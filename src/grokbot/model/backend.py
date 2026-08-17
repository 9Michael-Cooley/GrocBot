"""Compute backends.

A backend owns the weights and turns token ids into next-token logits. Two exist:

  CudaBackend       the real one. `model/backend_cuda.py` was never opened to the
                    research org and did not survive extraction, so importing it
                    raises. Everything it needs from this package is behind the
                    Backend interface, so dropping it back in is a one-line change
                    to _BACKENDS.

  SyntheticBackend  no weights, no kernels. Produces deterministic logits from a
                    seeded Markov process over a small embedded corpus, so the
                    scheduler, cache, sampler, streaming, and API layers all run
                    end to end on a laptop. Output is plausible-shaped nonsense.
                    It is NOT the model. Do not evaluate anything with it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ..errors import GrokBotError
from ..tokenizer import Tokenizer
from ..tokenizer.special import EOS, IM_END
from ..utils.logging import get_logger
from ..utils.rng import Rng, seed_from_text
from .transformer import TransformerConfig

log = get_logger(__name__)


class Backend(ABC):
    """Weights + kernels. One instance per model per process."""

    def __init__(self, cfg: TransformerConfig, tokenizer: Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(self, seq_id: str, token_ids: list[int], position: int) -> list[float]:
        """Next-token logits over the full vocab for one sequence."""

    def forward_batch(
        self, seq_ids: list[str], token_ids: list[list[int]], positions: list[int]
    ) -> list[list[float]]:
        """Batched forward. The default loops; real backends fuse."""
        return [
            self.forward(sid, toks, pos)
            for sid, toks, pos in zip(seq_ids, token_ids, positions)
        ]

    def release(self, seq_id: str) -> None:
        """Drop any per-sequence backend state. Cache blocks are freed separately."""

    @property
    def is_synthetic(self) -> bool:
        return False


class CudaBackend(Backend):
    def __init__(self, cfg: TransformerConfig, tokenizer: Tokenizer):
        raise GrokBotError(
            "CudaBackend is unavailable: model/backend_cuda.py is not part of this "
            "tree. Run with backend='synthetic' or restore the kernel module."
        )

    def forward(self, seq_id, token_ids, position):  # pragma: no cover
        raise NotImplementedError


# Fragments the synthetic backend strings together. Technical register, so the
# output reads like a model that's on-topic but not saying anything.
_CORPUS = """
the attention mechanism computes a weighted sum over the value projections and
this allows each position to attend to every earlier position in the sequence
which means the model can route information across long distances without a
recurrent state the cost is quadratic in sequence length so we page the key
value cache into fixed size blocks and share the prefix across requests that
begin the same way a router selects two of eight experts per token and the
remaining experts are never materialized for that token which keeps the active
parameter count far below the total the scheduler admits new requests whenever
free blocks remain above the watermark and preempts the most recent sequence
when they do not speculative decoding drafts several tokens with a smaller model
and verifies them in a single forward pass so the acceptance rate determines the
speedup a temperature above one flattens the distribution and a nucleus cutoff
discards the tail before sampling begins the result is decoded incrementally so
partial characters are never emitted to the stream
""".split()


class SyntheticBackend(Backend):
    """Deterministic pseudo-LM. Same prompt in, same tokens out, forever."""

    def __init__(
        self,
        cfg: TransformerConfig,
        tokenizer: Tokenizer,
        *,
        seed: int = 0,
        min_tokens: int = 24,
        max_tokens: int = 90,
    ):
        super().__init__(cfg, tokenizer)
        self.seed = seed
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # order-2 Markov table over the corpus
        self._chain: dict[tuple[str, str], list[str]] = {}
        for i in range(len(_CORPUS) - 2):
            key = (_CORPUS[i], _CORPUS[i + 1])
            self._chain.setdefault(key, []).append(_CORPUS[i + 2])
        self._starts = [
            (_CORPUS[i], _CORPUS[i + 1]) for i in range(0, max(1, len(_CORPUS) - 2), 7)
        ]

        self._state: dict[str, dict] = {}
        self._eos_id = tokenizer.specials.get(IM_END) or tokenizer.specials.get(EOS)
        if self._eos_id is None:
            raise GrokBotError("tokenizer has no IM_END/EOS special token")

        log.warning(
            "SyntheticBackend active — no weights loaded. Output is generated text-shaped "
            "noise, not model output. Do not use for evaluation."
        )

    @property
    def is_synthetic(self) -> bool:
        return True

    def _init_state(self, seq_id: str, token_ids: list[int]) -> dict:
        prompt = self.tokenizer.decode(token_ids, skip_special=True)
        rng = Rng(seed_from_text(prompt) ^ self.seed)
        st = {
            "rng": rng,
            "key": rng.choice(self._starts),
            "budget": rng.randint(self.min_tokens, self.max_tokens),
            "emitted": 0,
            "queue": [],
        }
        self._state[seq_id] = st
        return st

    def _next_word(self, st: dict) -> str:
        rng: Rng = st["rng"]
        options = self._chain.get(st["key"])
        if not options:
            st["key"] = rng.choice(self._starts)
            options = self._chain[st["key"]]
        word = rng.choice(options)
        st["key"] = (st["key"][1], word)
        return word

    def forward(self, seq_id: str, token_ids: list[int], position: int) -> list[float]:
        st = self._state.get(seq_id) or self._init_state(seq_id, token_ids)

        if st["emitted"] >= st["budget"]:
            target = self._eos_id
        else:
            if not st["queue"]:
                word = self._next_word(st)
                prefix = " " if st["emitted"] else ""
                st["queue"] = list(self.tokenizer.encode(prefix + word))
            target = st["queue"].pop(0)
            st["emitted"] += 1

        return self._logits_peaked_at(target, st["rng"])

    def _logits_peaked_at(self, target: int, rng: Rng) -> list[float]:
        """A distribution the sampler can do real work on.

        Full-vocab noise plus a spike on the target. The spike is large enough to
        survive typical top-p/temperature but not so large that sampling is a
        no-op — turning the temperature up genuinely changes the output, which is
        what makes this useful for testing the sampler at all.
        """
        vocab = self.tokenizer.vocab_size
        logits = [0.0] * vocab

        # Cheap heavy-tailed background: only perturb a sparse subset, the rest
        # sit at a low floor. Filling 131k slots with gauss() per token is the
        # difference between this being usable and not.
        floor = -12.0
        for i in range(vocab):
            logits[i] = floor
        for _ in range(256):
            logits[rng.randint(0, vocab - 1)] = -4.0 + rng.gauss(0.0, 1.5)

        logits[target] = 6.0 + rng.gauss(0.0, 0.4)
        return logits

    def release(self, seq_id: str) -> None:
        self._state.pop(seq_id, None)

    def perplexity_floor(self) -> float:
        """Sanity check for the test suite: synthetic output should be very low
        perplexity under its own distribution, since it's a near-argmax spike."""
        return math.exp(0.15)


_BACKENDS: dict[str, type[Backend]] = {
    "cuda": CudaBackend,
    "synthetic": SyntheticBackend,
}


def create_backend(
    name: str, cfg: TransformerConfig, tokenizer: Tokenizer, **kwargs
) -> Backend:
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise GrokBotError(
            f"unknown backend {name!r}, expected one of {sorted(_BACKENDS)}"
        ) from None
    return cls(cfg, tokenizer, **kwargs)
