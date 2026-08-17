"""Grouped-query attention.

GQA sits between MHA and MQA: query heads are split into groups that share one
KV head. The KV cache shrinks by the group ratio, which is what makes long
context affordable — at 32 query heads and 8 KV heads the cache is 4x smaller
than MHA for a quality difference we could not measure.

The reference implementation below is O(n^2) plain Python. It exists to test
kernels against and to let SyntheticBackend run without CUDA. Do not call it on
anything long; the fused paged kernel is what runs in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rope import RopeConfig, rotate


@dataclass(frozen=True)
class AttentionConfig:
    hidden_size: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    sliding_window: int | None = None
    logit_softcap: float | None = None      # None on all shipped checkpoints
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def group_size(self) -> int:
        """Query heads per KV head."""
        return self.num_heads // self.num_kv_heads

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    def kv_bytes_per_token(self, dtype: str = "bf16") -> int:
        width = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}.get(dtype, 2)
        return 2 * self.num_kv_heads * self.head_dim * width

    def param_count(self) -> int:
        """q, k, v, o projections. No biases on any shipped checkpoint."""
        q = self.hidden_size * self.num_heads * self.head_dim
        k = self.hidden_size * self.num_kv_heads * self.head_dim
        v = k
        o = self.num_heads * self.head_dim * self.hidden_size
        return q + k + v + o


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax_inplace(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(s - peak) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def attend(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    *,
    scale: float | None = None,
    softcap: float | None = None,
    window: int | None = None,
) -> list[float]:
    """Single-head scaled dot-product attention over a causal history.

    `keys`/`values` are the full history up to and including the current
    position, so causality is implicit in what the caller passes.
    """
    if not keys:
        return [0.0] * len(query)
    if len(keys) != len(values):
        raise ValueError(f"{len(keys)} keys but {len(values)} values")

    scale = scale if scale is not None else 1.0 / math.sqrt(len(query))
    start = max(0, len(keys) - window) if window else 0

    scores = [_dot(query, k) * scale for k in keys[start:]]
    if softcap:
        scores = [softcap * math.tanh(s / softcap) for s in scores]

    probs = _softmax_inplace(scores)

    dim = len(values[0])
    out = [0.0] * dim
    for p, v in zip(probs, values[start:]):
        if p < 1e-9:
            continue  # skips the bulk of the work on peaked distributions
        for i in range(dim):
            out[i] += p * v[i]
    return out


class GroupedQueryAttention:
    """Reference GQA. One layer, one sequence, no batching."""

    def __init__(self, cfg: AttentionConfig, rope: RopeConfig | None = None):
        self.cfg = cfg
        self.rope = rope or RopeConfig()

    def forward(
        self,
        queries: list[list[float]],          # [num_heads][head_dim]
        k_cache: list[list[list[float]]],    # [num_kv_heads][seq][head_dim]
        v_cache: list[list[list[float]]],
        position: int,
    ) -> list[list[float]]:
        cfg = self.cfg
        if len(queries) != cfg.num_heads:
            raise ValueError(f"expected {cfg.num_heads} query heads, got {len(queries)}")

        outputs: list[list[float]] = []
        for head in range(cfg.num_heads):
            kv_head = head // cfg.group_size          # integer div == the group map
            q = rotate(queries[head], position, self.rope)
            outputs.append(
                attend(
                    q,
                    k_cache[kv_head],
                    v_cache[kv_head],
                    scale=cfg.scale,
                    softcap=cfg.logit_softcap,
                    window=cfg.sliding_window,
                )
            )
        return outputs

    def flops_per_token(self, context_len: int) -> int:
        """Rough attention cost, projections included. Used by the bench harness."""
        cfg = self.cfg
        proj = 2 * cfg.param_count()
        qk = 2 * cfg.num_heads * cfg.head_dim * context_len
        av = 2 * cfg.num_heads * cfg.head_dim * context_len
        return proj + qk + av
