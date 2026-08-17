"""The engine. Ties config, backend, cache, scheduler, sampler, and streaming
into something you can call.

Single-threaded and synchronous on purpose. The step loop is the only place
sequence state mutates, which is what makes the scheduler debuggable at all; the
async surface in serve/ wraps this rather than reimplementing it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from ..config import Config
from ..errors import ContextLengthExceeded, RequestCancelled
from ..model.kv_cache import PagedKVCache
from ..model.loader import load_backend
from ..tokenizer.special import DEFAULT_STOPS, SUPPRESSED, render_chat
from ..utils.logging import get_logger
from .sampler import GenerationConfig, Sampler
from .scheduler import Request, Scheduler
from .stream import Completion, FinishReason, StreamAssembler, StreamChunk

log = get_logger(__name__)


class Engine:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.config = cfg

        self.backend, self.model_config, self.tokenizer = load_backend(cfg)

        block_size = cfg.get("cache.block_size", 16)
        num_blocks = cfg.get("cache.num_blocks", "auto")
        if num_blocks in ("auto", None):
            num_blocks = self._plan_blocks(block_size)

        kv_dtype = cfg.get("cache.dtype", "auto")
        if kv_dtype == "auto":
            kv_dtype = self.model_config.dtype

        self.cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            dtype=kv_dtype,
            enable_prefix_caching=cfg.get("cache.prefix_caching", True),
        )

        sched = cfg.section("scheduler")
        self.scheduler = Scheduler(
            self.cache,
            max_running=sched.get("max_running", 256),
            max_waiting=sched.get("max_waiting", 2048),
            max_batched_tokens=sched.get("max_batched_tokens", 8192),
            watermark=sched.get("watermark", 0.01),
            preemption=sched.get("preemption", "recompute"),
            chunked_prefill=sched.get("chunked_prefill", True),
            chunk_size=sched.get("chunk_size", 2048),
            max_context=self.model_config.max_context(),
        )

        self.sampler = Sampler(seed=cfg.get("runtime.seed", 0))
        self.default_gen = GenerationConfig.from_config(cfg.get("sampling", {}) or {})

        self._assemblers: dict[str, StreamAssembler] = {}
        self._suppressed = [
            self.tokenizer.specials[t] for t in SUPPRESSED if t in self.tokenizer.specials
        ]
        self._stop_ids = [
            self.tokenizer.specials[t] for t in DEFAULT_STOPS if t in self.tokenizer.specials
        ]

        log.info(
            "engine ready: %s, %d cache blocks (%.1f GiB), backend=%s",
            self.model_config.name,
            self.cache.num_blocks,
            self.cache.num_blocks * self.cache.bytes_per_block() / 2**30,
            "synthetic" if self.backend.is_synthetic else "cuda",
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path, **overrides) -> Engine:
        cfg = Config.load(path)
        if overrides:
            cfg.apply_overrides(overrides)
        return cls(cfg)

    def _plan_blocks(self, block_size: int) -> int:
        """Size the cache. Without a real device we can't query free memory, so
        the synthetic path takes a small fixed pool — enough for the test suite,
        small enough that eviction and preemption actually get exercised."""
        if self.backend.is_synthetic:
            return 2048
        total = int(self.config.get("runtime.device_memory_bytes", 80 * 2**30))
        return self.model_config.plan_kv_blocks(
            total,
            block_size=block_size,
            utilization=self.config.get("runtime.gpu_memory_utilization", 0.9),
            kv_dtype=self.config.get("cache.dtype") if self.config.get("cache.dtype") != "auto" else None,
            tensor_parallel=self.config.get("runtime.tensor_parallel", 1),
        )

    # -- public API --------------------------------------------------------

    def generate(self, prompt: str, gen: GenerationConfig | None = None) -> Completion:
        chunks = list(self.stream(prompt, gen))
        req_id = self._last_request_id
        assembler = self._assemblers.pop(req_id, None)
        reason = chunks[-1].finish_reason if chunks else FinishReason.ERROR
        if assembler is None:
            return Completion(finish_reason=reason)
        return assembler.completion(
            reason or FinishReason.LENGTH,
            prompt_tokens=self._last_prompt_tokens,
            cached=self._last_cached_tokens,
        )

    def chat(self, messages: list[dict], gen: GenerationConfig | None = None) -> Completion:
        return self.generate(render_chat(messages), gen)

    def stream_chat(
        self, messages: list[dict], gen: GenerationConfig | None = None
    ) -> Iterator[StreamChunk]:
        yield from self.stream(render_chat(messages), gen)

    def stream(self, prompt: str, gen: GenerationConfig | None = None) -> Iterator[StreamChunk]:
        cfg = gen or self.default_gen
        cfg = cfg.merged(suppress_tokens=list(cfg.suppress_tokens) + self._suppressed)
        if cfg.seed is not None:
            self.sampler.reseed(cfg.seed)

        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("empty prompt")

        limit = self.model_config.max_context()
        if len(token_ids) + cfg.max_tokens > limit:
            raise ContextLengthExceeded(
                f"prompt {len(token_ids)} + max_tokens {cfg.max_tokens} exceeds context {limit}"
            )

        request = Request(prompt_token_ids=token_ids, gen_config=cfg)
        self._last_request_id = request.request_id
        self._last_prompt_tokens = len(token_ids)
        self._last_cached_tokens = 0

        self.scheduler.add_request(request)
        self._assemblers[request.request_id] = StreamAssembler(self.tokenizer, cfg)

        try:
            while self.scheduler.has_work:
                for chunk in self._step():
                    yield chunk
                    if chunk.is_final:
                        return
        finally:
            self.backend.release(request.request_id)

    # -- the loop ----------------------------------------------------------

    def _step(self) -> list[StreamChunk]:
        output = self.scheduler.schedule()
        emitted: list[StreamChunk] = []

        for req in output.preempted:
            # Recompute preemption discards decoded KV but keeps the text we
            # already streamed. The assembler is intentionally not reset.
            log.debug("preempted %s (attempt %d)", req.request_id, req.preemption_count)

        # prefill chunks
        for req in output.prefill:
            chunk = output.chunk_sizes[req.request_id]
            for i in range(req.prefill_offset, min(req.prompt_len, req.prefill_offset + chunk)):
                if i >= self.cache.get_table(req.request_id).num_tokens:
                    self.cache.append_token(req.request_id)
            self.scheduler.advance_prefill(req, chunk)
            if req.request_id == getattr(self, "_last_request_id", None):
                self._last_cached_tokens = req.cached_tokens

        # decodes
        ready = [r for r in output.decode if r.prefill_done]
        ready += [r for r in output.prefill if r.prefill_done and r not in output.decode]
        if not ready:
            return emitted

        logits_batch = self.backend.forward_batch(
            [r.request_id for r in ready],
            [r.all_token_ids for r in ready],
            [r.total_len - 1 for r in ready],
        )

        for req, logits in zip(ready, logits_batch):
            token_id, logprob = self.sampler.sample(logits, req.all_token_ids, req.gen_config)
            self.scheduler.append_token(req, token_id)

            assembler = self._assemblers.get(req.request_id)
            if assembler is None:
                continue

            reason = self._finish_reason(req, token_id, assembler)
            if reason is None:
                chunk = assembler.push(token_id, logprob)
                if chunk is not None:
                    emitted.append(chunk)
                    if chunk.is_final:      # stop string landed mid-token
                        self.scheduler.finish(req)
                continue

            if reason is FinishReason.STOP_TOKEN:
                pass  # don't render the stop token itself
            else:
                chunk = assembler.push(token_id, logprob)
                if chunk is not None and chunk.text:
                    emitted.append(chunk)

            emitted.append(assembler.finalize(reason))
            self.scheduler.finish(req)

        return emitted

    def _finish_reason(
        self, req: Request, token_id: int, assembler: StreamAssembler
    ) -> FinishReason | None:
        cfg = req.gen_config
        if token_id in self._stop_ids or token_id in cfg.stop_token_ids:
            return FinishReason.STOP_TOKEN
        if req.num_generated >= cfg.max_tokens:
            return FinishReason.LENGTH
        if assembler.stop_checker.stopped:
            return FinishReason.STOP
        return None

    # -- lifecycle ---------------------------------------------------------

    def abort(self, request_id: str) -> bool:
        self._assemblers.pop(request_id, None)
        self.backend.release(request_id)
        return self.scheduler.abort(request_id)

    def reset(self) -> None:
        self.cache.reset()
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self._assemblers.clear()

    def stats(self) -> dict:
        return {
            "model": self.model_config.name,
            "backend": "synthetic" if self.backend.is_synthetic else "cuda",
            "scheduler": self.scheduler.snapshot(),
        }

    def benchmark_step(self, num_requests: int, prompt_tokens: int, output_tokens: int) -> dict:
        """Drive a synthetic load through the scheduler. Used by benchmarks/."""
        gen = self.default_gen.merged(max_tokens=output_tokens, temperature=0.0)
        start = time.monotonic()
        for i in range(num_requests):
            ids = self.tokenizer.encode(f"benchmark request {i} " * max(1, prompt_tokens // 4))
            self.scheduler.add_request(
                Request(prompt_token_ids=ids[:prompt_tokens] or ids[:1], gen_config=gen)
            )
            self._assemblers[self.scheduler.waiting[-1].request_id] = StreamAssembler(
                self.tokenizer, gen
            )
        steps = 0
        while self.scheduler.has_work:
            self._step()
            steps += 1
            if steps > num_requests * output_tokens * 4:
                raise RequestCancelled("benchmark did not converge; scheduler is stuck")
        elapsed = time.monotonic() - start
        snap = self.scheduler.snapshot()
        return {
            "elapsed_s": elapsed,
            "steps": steps,
            "requests": num_requests,
            "decode_tokens": snap["decode_tokens"],
            "prefill_tokens": snap["prefill_tokens"],
            "decode_tok_s": snap["decode_tokens"] / elapsed if elapsed else 0.0,
            "preemptions": snap["preemptions"],
        }
