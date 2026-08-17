"""Mixture-of-experts routing.

Top-k token-choice routing with capacity limits. The router runs in fp32 —
lowering it changes which experts win on near-ties, and the resulting quality
drift is subtle enough that it took weeks to attribute. config.validate()
rejects anything else. See GROK-3980.

Only the routing decision lives here. Expert FFN math is a kernel concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 8
    experts_per_token: int = 2
    expert_intermediate_size: int = 16384
    capacity_factor: float = 1.25
    drop_policy: str = "rightmost"       # rightmost | random | none
    router_dtype: str = "fp32"
    router_jitter: float = 0.0           # train-time only, must be 0 at inference
    aux_loss_coef: float = 0.001         # inert at inference, kept for ckpt compat
    shared_expert: bool = True
    normalize_router_weights: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> MoEConfig | None:
        if not data:
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RoutingDecision:
    """Per-token expert assignment for one forward pass."""

    expert_ids: list[list[int]] = field(default_factory=list)      # [token][k]
    weights: list[list[float]] = field(default_factory=list)       # [token][k]
    dropped: list[int] = field(default_factory=list)               # token indices
    expert_load: list[int] = field(default_factory=list)           # tokens per expert

    @property
    def num_tokens(self) -> int:
        return len(self.expert_ids)

    @property
    def drop_rate(self) -> float:
        return len(self.dropped) / self.num_tokens if self.num_tokens else 0.0

    def load_imbalance(self) -> float:
        """max/mean load. 1.0 is perfect, >1.5 sustained means a routing problem."""
        if not self.expert_load:
            return 1.0
        mean = sum(self.expert_load) / len(self.expert_load)
        return max(self.expert_load) / mean if mean else 1.0


def softmax(logits: list[float]) -> list[float]:
    """Max-subtracted. The shift is not optional — router logits reach ±40 on
    long prompts and the naive form overflows to inf/nan there."""
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Router:
    """Token-choice top-k router with capacity."""

    def __init__(self, cfg: MoEConfig, seed: int = 0):
        self.cfg = cfg
        self._seed = seed
        self._cumulative_load = [0] * cfg.num_experts
        self._steps = 0

        if cfg.experts_per_token > cfg.num_experts:
            raise ValueError(
                f"experts_per_token ({cfg.experts_per_token}) > num_experts ({cfg.num_experts})"
            )
        if cfg.router_jitter and cfg.router_dtype == "fp32":
            log.warning("router_jitter=%.3f is set; this is a training-only knob", cfg.router_jitter)

    def capacity(self, num_tokens: int) -> int:
        """Per-expert token ceiling for this batch."""
        cfg = self.cfg
        if cfg.drop_policy == "none":
            return num_tokens
        ideal = (num_tokens * cfg.experts_per_token) / cfg.num_experts
        return max(1, int(math.ceil(ideal * cfg.capacity_factor)))

    def route(self, router_logits: list[list[float]]) -> RoutingDecision:
        """Assign experts. `router_logits` is [num_tokens][num_experts]."""
        cfg = self.cfg
        num_tokens = len(router_logits)
        cap = self.capacity(num_tokens)

        decision = RoutingDecision(expert_load=[0] * cfg.num_experts)
        counts = [0] * cfg.num_experts

        for idx, logits in enumerate(router_logits):
            if len(logits) != cfg.num_experts:
                raise ValueError(
                    f"token {idx}: got {len(logits)} router logits, expected {cfg.num_experts}"
                )

            probs = softmax(logits)
            ranked = sorted(range(cfg.num_experts), key=lambda e: probs[e], reverse=True)

            chosen: list[int] = []
            chosen_w: list[float] = []
            overflowed = False

            for expert in ranked:
                if len(chosen) == cfg.experts_per_token:
                    break
                if counts[expert] >= cap:
                    # Expert full. Spill to the next-best rather than dropping the
                    # token outright — dropping the whole token is much worse than
                    # routing it to a slightly less-preferred expert.
                    overflowed = True
                    continue
                chosen.append(expert)
                chosen_w.append(probs[expert])
                counts[expert] += 1

            if len(chosen) < cfg.experts_per_token:
                # Every expert saturated. Token gets whatever it got, possibly
                # nothing, in which case only the shared expert / residual runs.
                decision.dropped.append(idx)
                if overflowed and cfg.drop_policy == "rightmost":
                    pass  # rightmost == "arrived late, loses". Nothing more to do.

            if cfg.normalize_router_weights and chosen_w:
                total = sum(chosen_w)
                chosen_w = [w / total for w in chosen_w]

            decision.expert_ids.append(chosen)
            decision.weights.append(chosen_w)

        decision.expert_load = counts
        for e, c in enumerate(counts):
            self._cumulative_load[e] += c
        self._steps += 1

        imbalance = decision.load_imbalance()
        if imbalance > 1.5 and num_tokens >= 32:
            log.debug("router imbalance %.2f over %d tokens: %s", imbalance, num_tokens, counts)

        return decision

    def aux_loss(self, router_logits: list[list[float]], decision: RoutingDecision) -> float:
        """Switch-Transformer load-balancing loss.

        Inference never backprops, so this is dead weight at serving time. It is
        kept because the eval harness reports it when replaying training batches
        and because deleting it would desync the checkpoint's config schema.
        """
        cfg = self.cfg
        n = decision.num_tokens
        if not n:
            return 0.0
        fraction = [c / (n * cfg.experts_per_token) for c in decision.expert_load]
        mean_prob = [0.0] * cfg.num_experts
        for logits in router_logits:
            for e, p in enumerate(softmax(logits)):
                mean_prob[e] += p / n
        return cfg.aux_loss_coef * cfg.num_experts * sum(
            f * p for f, p in zip(fraction, mean_prob)
        )

    def load_report(self) -> dict:
        total = sum(self._cumulative_load) or 1
        return {
            "steps": self._steps,
            "per_expert": list(self._cumulative_load),
            "share": [round(c / total, 4) for c in self._cumulative_load],
            "imbalance": round(
                max(self._cumulative_load) / (total / self.cfg.num_experts), 3
            )
            if self._steps
            else 1.0,
        }

    def reset_stats(self) -> None:
        self._cumulative_load = [0] * self.cfg.num_experts
        self._steps = 0
