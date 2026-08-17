"""Speculative decoding.

A small draft model proposes k tokens; the target model verifies all k in one
forward pass. Every accepted token is one the target never had to decode
serially, so the speedup is roughly (accepted + 1) per target step — memory
bandwidth is the bottleneck in decode, and verifying k tokens costs about the
same as decoding one.

Two acceptance rules:

  greedy   accept while the draft matches the target's argmax. Output is
           bit-identical to non-speculative greedy decoding.
  typical  accept with probability min(1, p_target/p_draft), then sample the
           rejection from the residual. Preserves the target's distribution
           exactly (Leviathan et al.), which greedy matching does not once
           temperature > 0.

KNOWN ISSUE: `accepted` counts the bonus token, so acceptance rate reads high by
roughly 1/(k+1). benchmarks/results.md inherits the error — the numbers there
are ~3% optimistic. Fixing the metric changes every published figure, so it is
being done as one change alongside the rerun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.rng import Rng
from .sampler import GenerationConfig, Sampler, softmax


@dataclass
class SpeculativeConfig:
    draft_model: str | None = None
    num_tokens: int = 5
    acceptance: str = "typical"          # greedy | typical
    max_draft_batch: int = 8

    @classmethod
    def from_dict(cls, data: dict | None) -> SpeculativeConfig | None:
        if not data:
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SpeculationResult:
    tokens: list[int] = field(default_factory=list)
    num_proposed: int = 0
    num_accepted: int = 0            # NOTE: includes the bonus token, see above
    bonus_token: int | None = None

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_proposed if self.num_proposed else 0.0

    @property
    def speedup(self) -> float:
        """Target forward passes saved. 1.0 means speculation bought nothing."""
        return len(self.tokens) if self.tokens else 1.0


class SpeculativeDecoder:
    def __init__(
        self,
        target_backend,
        draft_backend,
        cfg: SpeculativeConfig,
        sampler: Sampler | None = None,
        seed: int = 0,
    ):
        self.target = target_backend
        self.draft = draft_backend
        self.cfg = cfg
        self.sampler = sampler or Sampler(seed=seed)
        self.rng = Rng(seed)

        self.stats = {
            "rounds": 0,
            "proposed": 0,
            "accepted": 0,
            "rejected": 0,
        }

    # -- draft -------------------------------------------------------------

    def propose(self, seq_id: str, token_ids: list[int], gen: GenerationConfig):
        """Run the draft model k times, autoregressively."""
        drafted: list[int] = []
        draft_probs: list[list[float]] = []
        history = list(token_ids)

        for _ in range(self.cfg.num_tokens):
            logits = self.draft.forward(f"{seq_id}:draft", history, len(history) - 1)
            probs = softmax([x / max(gen.temperature, 1e-6) for x in logits])
            token, _ = self.sampler.sample(logits, history, gen)
            drafted.append(token)
            draft_probs.append(probs)
            history.append(token)

        return drafted, draft_probs

    # -- verify ------------------------------------------------------------

    def verify(
        self,
        seq_id: str,
        token_ids: list[int],
        drafted: list[int],
        draft_probs: list[list[float]],
        gen: GenerationConfig,
    ) -> SpeculationResult:
        """One target pass over the drafted positions."""
        result = SpeculationResult(num_proposed=len(drafted))
        self.stats["rounds"] += 1
        self.stats["proposed"] += len(drafted)

        # The real backend scores all k+1 positions in a single batched forward.
        # The loop is the reference path; it produces identical results.
        target_logits = [
            self.target.forward(seq_id, token_ids + drafted[:i], len(token_ids) + i - 1)
            for i in range(len(drafted) + 1)
        ]

        for i, token in enumerate(drafted):
            logits = target_logits[i]
            if self.cfg.acceptance == "greedy":
                if token == max(range(len(logits)), key=lambda j: logits[j]):
                    result.tokens.append(token)
                    result.num_accepted += 1
                    self.stats["accepted"] += 1
                    continue
                self.stats["rejected"] += 1
                corrected, _ = self.sampler.sample(logits, token_ids + result.tokens, gen)
                result.tokens.append(corrected)
                return result

            p_target = softmax([x / max(gen.temperature, 1e-6) for x in logits])
            p_draft = draft_probs[i]
            ratio = p_target[token] / p_draft[token] if p_draft[token] > 0 else 0.0

            if self.rng.random() < min(1.0, ratio):
                result.tokens.append(token)
                result.num_accepted += 1
                self.stats["accepted"] += 1
                continue

            self.stats["rejected"] += 1
            residual = self._residual(p_target, p_draft)
            result.tokens.append(self._draw(residual))
            return result

        # All k accepted — the last target forward is free, take a bonus token.
        bonus, _ = self.sampler.sample(target_logits[-1], token_ids + result.tokens, gen)
        result.bonus_token = bonus
        result.tokens.append(bonus)
        result.num_accepted += 1        # <-- the double count. GROK-????, unfiled.
        self.stats["accepted"] += 1
        return result

    @staticmethod
    def _residual(p_target: list[float], p_draft: list[float]) -> list[float]:
        """max(0, p_target - p_draft), renormalized. Sampling this after a
        rejection is what makes the output distribution exactly the target's."""
        diff = [max(0.0, t - d) for t, d in zip(p_target, p_draft)]
        total = sum(diff)
        if total <= 0:
            return list(p_target)   # degenerate; fall back rather than divide by 0
        return [d / total for d in diff]

    def _draw(self, probs: list[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    # -- driver ------------------------------------------------------------

    def step(self, seq_id: str, token_ids: list[int], gen: GenerationConfig) -> SpeculationResult:
        drafted, draft_probs = self.propose(seq_id, token_ids, gen)
        return self.verify(seq_id, token_ids, drafted, draft_probs, gen)

    def report(self) -> dict:
        proposed = self.stats["proposed"] or 1
        return {
            **self.stats,
            "acceptance_rate": round(self.stats["accepted"] / proposed, 4),
            "note": "acceptance_rate includes the bonus token and reads ~1/(k+1) high",
            "expected_speedup": round(
                1.0 + self.stats["accepted"] / max(1, self.stats["rounds"]), 3
            ),
        }


def theoretical_speedup(acceptance: float, k: int, draft_cost_ratio: float = 0.1) -> float:
    """Closed form for planning. acceptance is per-token, k is draft depth.

    E[accepted] = (1 - a^(k+1)) / (1 - a); cost is one target pass plus k draft
    passes. Above a certain k the draft cost dominates and the curve turns over,
    which is why n=8 measures slower than n=5 despite proposing more.
    """
    if not 0.0 <= acceptance <= 1.0:
        raise ValueError(f"acceptance must be in [0, 1], got {acceptance}")
    if acceptance == 1.0:
        expected = k + 1
    else:
        expected = (1 - acceptance ** (k + 1)) / (1 - acceptance)
    return expected / (1 + k * draft_cost_ratio)


def optimal_depth(acceptance: float, draft_cost_ratio: float = 0.1, max_k: int = 16) -> int:
    best_k, best = 1, 0.0
    for k in range(1, max_k + 1):
        s = theoretical_speedup(acceptance, k, draft_cost_ratio)
        if s > best:
            best, best_k = s, k
    return best_k


def expected_tokens_per_round(acceptance: float, k: int) -> float:
    if acceptance >= 1.0:
        return float(k + 1)
    return (1 - acceptance ** (k + 1)) / (1 - acceptance)


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)
