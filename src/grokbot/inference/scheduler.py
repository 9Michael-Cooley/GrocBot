"""Continuous-batching scheduler.

Requests join the running batch the moment a slot frees rather than waiting for
the whole batch to finish, so short requests aren't held hostage by long ones.
Each step builds a batch under two budgets: token budget (max_batched_tokens)
and cache budget (free blocks above the watermark).

Prefill and decode are mixed in one step when chunked_prefill is on. A prefill
of 100k tokens is split into chunks so it can't stall every decoding sequence
behind it — that head-of-line block was the single worst p99 contributor before
chunking landed.

KNOWN ISSUE (GROK-4417): preemption is LIFO. Above ~400 concurrent with long
prompts it thrashes — the victim is re-admitted, preempts someone else, and the
recompute cost cascades. Workaround is --max-running 384. The fix is a priority
queue keyed on (arrival, tokens_done) so we stop evicting sequences that are
nearly finished.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from ..errors import ContextLengthExceeded, OutOfCacheBlocks
from ..model.kv_cache import PagedKVCache
from ..utils.logging import get_logger
from .sampler import GenerationConfig

log = get_logger(__name__)
_counter = itertools.count()


class SeqState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class Request:
    prompt_token_ids: list[int]
    gen_config: GenerationConfig
    request_id: str = ""
    arrival: float = field(default_factory=time.monotonic)
    state: SeqState = SeqState.WAITING

    output_token_ids: list[int] = field(default_factory=list)
    prefill_offset: int = 0          # tokens of the prompt already processed
    cached_tokens: int = 0
    preemption_count: int = 0
    first_token_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = f"req-{next(_counter):08d}"

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def prefill_done(self) -> bool:
        return self.prefill_offset >= self.prompt_len

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def ttft(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.arrival

    def latency(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.arrival


@dataclass
class SchedulerOutput:
    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    chunk_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def num_sequences(self) -> int:
        return len(self.prefill) + len(self.decode)


class Scheduler:
    def __init__(
        self,
        cache: PagedKVCache,
        *,
        max_running: int = 256,
        max_waiting: int = 2048,
        max_batched_tokens: int = 8192,
        watermark: float = 0.01,
        preemption: str = "recompute",
        chunked_prefill: bool = True,
        chunk_size: int = 2048,
        max_context: int = 131072,
    ):
        self.cache = cache
        self.max_running = max_running
        self.max_waiting = max_waiting
        self.max_batched_tokens = max_batched_tokens
        self.watermark = watermark
        self.preemption = preemption
        self.chunked_prefill = chunked_prefill
        self.chunk_size = chunk_size
        self.max_context = max_context

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []

        self.stats = {
            "admitted": 0,
            "completed": 0,
            "preemptions": 0,
            "steps": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
        }

        if preemption == "swap":
            # swap-out to host memory was never finished; the allocator has no
            # concept of a swapped block. Fail loudly rather than half-work.
            raise NotImplementedError(
                "preemption='swap' is unfinished; use 'recompute'"
            )

    # -- admission ---------------------------------------------------------

    def add_request(self, request: Request) -> None:
        budget = request.prompt_len + request.gen_config.max_tokens
        if budget > self.max_context:
            raise ContextLengthExceeded(
                f"{request.request_id}: prompt {request.prompt_len} + max_tokens "
                f"{request.gen_config.max_tokens} = {budget} exceeds context {self.max_context}"
            )
        if len(self.waiting) >= self.max_waiting:
            raise OutOfCacheBlocks(
                f"waiting queue full ({self.max_waiting}); shed load upstream"
            )
        self.waiting.append(request)

    def abort(self, request_id: str) -> bool:
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if req.request_id == request_id:
                    queue.remove(req)
                    req.state = SeqState.FINISHED
                    req.finished_at = time.monotonic()
                    self.cache.free(req.request_id)
                    return True
        return False

    # -- the step ----------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        self.stats["steps"] += 1
        out = SchedulerOutput()
        token_budget = self.max_batched_tokens

        # 1. Decodes first. They already hold their blocks and each needs exactly
        #    one new slot, so they're cheap and starving them is what users feel.
        for req in list(self.running):
            if not req.prefill_done:
                continue
            if token_budget < 1:
                break
            if not self.cache.can_allocate(1, watermark=self.watermark):
                victim = self._preempt()
                if victim is None:
                    break
                out.preempted.append(victim)
                if victim is req:
                    continue
            out.decode.append(req)
            token_budget -= 1

        # 2. Continue in-flight prefills before admitting new ones. A partially
        #    prefilled sequence is already holding blocks; finishing it releases
        #    the batch slot sooner than starting something new.
        for req in list(self.running):
            if req.prefill_done or token_budget <= 0:
                continue
            chunk = self._plan_chunk(req, token_budget)
            if chunk <= 0:
                continue
            out.prefill.append(req)
            out.chunk_sizes[req.request_id] = chunk
            token_budget -= chunk

        # 3. Admit from the waiting queue, FIFO.
        while self.waiting and token_budget > 0 and len(self.running) < self.max_running:
            req = self.waiting[0]
            chunk = self._plan_chunk(req, token_budget)
            if chunk <= 0:
                break

            need = chunk if self.chunked_prefill else req.prompt_len
            if not self.cache.can_allocate(need, watermark=self.watermark):
                victim = self._preempt()
                if victim is None:
                    break          # nothing left to evict; try again next step
                out.preempted.append(victim)
                continue

            self.waiting.pop(0)
            try:
                table = self.cache.allocate(req.request_id, req.prompt_token_ids[:chunk],
                                            watermark=self.watermark)
            except OutOfCacheBlocks:
                self.waiting.insert(0, req)   # put it back, don't lose it
                break

            req.cached_tokens = table.num_cached
            req.state = SeqState.PREFILL
            self.running.append(req)
            self.stats["admitted"] += 1

            out.prefill.append(req)
            out.chunk_sizes[req.request_id] = chunk
            token_budget -= chunk

        self.stats["prefill_tokens"] += sum(out.chunk_sizes.values())
        self.stats["decode_tokens"] += len(out.decode)
        return out

    def _plan_chunk(self, req: Request, budget: int) -> int:
        remaining = req.prompt_len - req.prefill_offset
        if remaining <= 0:
            return 0
        if not self.chunked_prefill:
            return remaining if remaining <= budget else 0
        return min(remaining, self.chunk_size, budget)

    # -- preemption --------------------------------------------------------

    def _preempt(self) -> Request | None:
        """Evict to free blocks.

        LIFO — most recently admitted loses. See GROK-4417; this is the
        thrashing source. Deliberately not "fixed" locally because the queue
        change touches the admission path and needs its own rollout.
        """
        if not self.running:
            return None

        victim = self.running[-1]
        self.running.pop()
        self.cache.free(victim.request_id)

        victim.state = SeqState.PREEMPTED
        victim.preemption_count += 1
        # Recompute: throw away prefill progress but keep generated tokens, and
        # fold them into the prompt so the retry regenerates the same KV state.
        victim.prompt_token_ids = victim.all_token_ids
        victim.output_token_ids = []
        victim.prefill_offset = 0

        self.waiting.insert(0, victim)   # front of the queue; it's already old
        self.stats["preemptions"] += 1

        if victim.preemption_count == 5:
            log.warning(
                "%s preempted %d times — scheduler is thrashing (GROK-4417)",
                victim.request_id,
                victim.preemption_count,
            )
        return victim

    # -- completion --------------------------------------------------------

    def advance_prefill(self, req: Request, chunk: int) -> None:
        req.prefill_offset = min(req.prompt_len, req.prefill_offset + chunk)
        if req.prefill_done:
            req.state = SeqState.RUNNING

    def append_token(self, req: Request, token_id: int) -> None:
        if req.first_token_at is None:
            req.first_token_at = time.monotonic()
        req.output_token_ids.append(token_id)
        self.cache.append_token(req.request_id)

    def finish(self, req: Request) -> None:
        if req in self.running:
            self.running.remove(req)
        req.state = SeqState.FINISHED
        req.finished_at = time.monotonic()
        self.cache.free(req.request_id)
        self.finished.append(req)
        self.stats["completed"] += 1

    # -- introspection -----------------------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def snapshot(self) -> dict:
        return {
            **self.stats,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "cache": self.cache.snapshot(),
        }
