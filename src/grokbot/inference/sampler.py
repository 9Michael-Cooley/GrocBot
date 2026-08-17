"""Token sampling.

Order matters and is not arbitrary:

    logit bias -> suppression -> penalties -> temperature -> top-k -> top-p
    -> min-p -> renormalize -> draw

Penalties go before temperature so that turning the temperature up doesn't
quietly rescale how hard the penalty bites. Truncation goes after temperature so
top-p measures the distribution the user actually asked for. Getting this order
wrong produces output that is subtly worse in ways no test catches, so don't
reorder it without a side-by-side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.rng import Rng


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.0
    repetition_penalty: float = 1.0      # multiplicative, applied to seen tokens
    frequency_penalty: float = 0.0       # additive, scales with count
    presence_penalty: float = 0.0        # additive, flat once seen
    penalty_window: int = 256            # only the last N tokens count
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    suppress_tokens: list[int] = field(default_factory=list)
    logit_bias: dict[int, float] = field(default_factory=dict)
    seed: int | None = None
    n: int = 1
    echo_reasoning: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.min_p and not 0.0 <= self.min_p < 1.0:
            raise ValueError(f"min_p must be in [0, 1), got {self.min_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @classmethod
    def from_config(cls, section: dict) -> GenerationConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (section or {}).items() if k in known and v is not None})

    def merged(self, **overrides) -> GenerationConfig:
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return GenerationConfig(**data)


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Sampler:
    """Stateless w.r.t. sequences; the RNG is the only carried state."""

    def __init__(self, seed: int = 0):
        self.rng = Rng(seed)

    # -- individual steps --------------------------------------------------

    @staticmethod
    def apply_logit_bias(logits: list[float], bias: dict[int, float]) -> None:
        for tid, delta in bias.items():
            if 0 <= tid < len(logits):
                logits[tid] += delta

    @staticmethod
    def suppress(logits: list[float], token_ids: list[int]) -> None:
        for tid in token_ids:
            if 0 <= tid < len(logits):
                logits[tid] = -math.inf

    @staticmethod
    def apply_penalties(logits: list[float], history: list[int], cfg: GenerationConfig) -> None:
        if cfg.repetition_penalty == 1.0 and not cfg.frequency_penalty and not cfg.presence_penalty:
            return

        window = history[-cfg.penalty_window :] if cfg.penalty_window else history
        counts: dict[int, int] = {}
        for tid in window:
            counts[tid] = counts.get(tid, 0) + 1

        for tid, count in counts.items():
            if not 0 <= tid < len(logits):
                continue
            if cfg.repetition_penalty != 1.0:
                # Divide positive logits, multiply negative ones. Multiplying
                # throughout would *reward* tokens whose logit is negative.
                if logits[tid] > 0:
                    logits[tid] /= cfg.repetition_penalty
                else:
                    logits[tid] *= cfg.repetition_penalty
            if cfg.frequency_penalty:
                logits[tid] -= cfg.frequency_penalty * count
            if cfg.presence_penalty:
                logits[tid] -= cfg.presence_penalty

    @staticmethod
    def apply_temperature(logits: list[float], temperature: float) -> None:
        if temperature in (0.0, 1.0):
            return
        for i in range(len(logits)):
            logits[i] /= temperature

    @staticmethod
    def top_k_filter(logits: list[float], k: int) -> None:
        if k <= 0 or k >= len(logits):
            return
        threshold = sorted(logits, reverse=True)[k - 1]
        for i, v in enumerate(logits):
            if v < threshold:
                logits[i] = -math.inf

    @staticmethod
    def top_p_filter(logits: list[float], p: float) -> None:
        """Nucleus. Always keeps at least one token, even if p is tiny."""
        if p >= 1.0:
            return
        probs = softmax(logits)
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        cumulative = 0.0
        keep: set[int] = set()
        for idx in order:
            keep.add(idx)
            cumulative += probs[idx]
            if cumulative >= p:
                break
        for i in range(len(logits)):
            if i not in keep:
                logits[i] = -math.inf

    @staticmethod
    def min_p_filter(logits: list[float], min_p: float) -> None:
        """Drop tokens under min_p * p_max. Scales the cutoff with confidence,
        unlike top-p — on a peaked step it prunes harder, on a flat one it barely
        prunes at all."""
        if min_p <= 0.0:
            return
        probs = softmax(logits)
        cutoff = min_p * max(probs)
        for i, p in enumerate(probs):
            if p < cutoff:
                logits[i] = -math.inf

    # -- draw --------------------------------------------------------------

    def _multinomial(self, probs: list[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        # Float error can leave acc a hair under 1.0; fall back to the last
        # token with nonzero mass rather than returning 0, which would silently
        # bias toward token id 0.
        for i in range(len(probs) - 1, -1, -1):
            if probs[i] > 0:
                return i
        raise RuntimeError("all-zero probability vector")

    def sample(
        self, logits: list[float], history: list[int], cfg: GenerationConfig
    ) -> tuple[int, float]:
        """Returns (token_id, logprob-under-the-final-distribution)."""
        if not logits:
            raise ValueError("empty logit vector")

        work = list(logits)

        if cfg.logit_bias:
            self.apply_logit_bias(work, cfg.logit_bias)
        if cfg.suppress_tokens:
            self.suppress(work, cfg.suppress_tokens)
        self.apply_penalties(work, history, cfg)

        if cfg.is_greedy:
            best = max(range(len(work)), key=lambda i: work[i])
            return best, math.log(max(softmax(work)[best], 1e-12))

        self.apply_temperature(work, cfg.temperature)
        self.top_k_filter(work, cfg.top_k)
        self.top_p_filter(work, cfg.top_p)
        self.min_p_filter(work, cfg.min_p)

        probs = softmax(work)
        token = self._multinomial(probs)
        return token, math.log(max(probs[token], 1e-12))

    def top_logprobs(
        self, logits: list[float], k: int, cfg: GenerationConfig
    ) -> list[tuple[int, float]]:
        """Top-k (id, logprob) before truncation. The API layer accepts a
        logprobs field and currently ignores it; this is what it would call."""
        work = list(logits)
        self.apply_temperature(work, cfg.temperature)
        probs = softmax(work)
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
        return [(i, math.log(max(probs[i], 1e-12))) for i in order]

    def reseed(self, seed: int) -> None:
        self.rng = Rng(seed)
