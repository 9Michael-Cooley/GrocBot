"""Rotary position embeddings.

Standard RoPE plus the two extrapolation schemes we ship: linear (position
interpolation) and YaRN. YaRN is what the long-context checkpoints use — it
interpolates low-frequency dimensions while leaving high-frequency ones alone,
so local structure survives the stretch in a way plain PI does not.

The math here is real and runs on plain floats. The fused kernel that actually
executes at serving time lives in the CUDA backend; this is the reference the
kernel is tested against, and it is what SyntheticBackend uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RopeConfig:
    theta: float = 10000.0
    scaling: str | None = None       # None | "linear" | "yarn"
    factor: float = 1.0
    original_context: int = 8192
    beta_fast: int = 32              # yarn: dims rotating faster than this are untouched
    beta_slow: int = 1               # yarn: dims slower than this are fully interpolated
    mscale: float = 1.0              # attention-entropy correction, 0 => derive from factor

    @classmethod
    def from_dict(cls, data: dict | None) -> RopeConfig:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _yarn_correction_dim(num_rotations: float, dim: int, theta: float, original: int) -> float:
    """Which dimension index completes `num_rotations` over the original context."""
    return (dim * math.log(original / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def _yarn_ramp(low: float, high: float, dim: int) -> list[float]:
    """Linear ramp in [0,1] over [low, high]; clamped outside."""
    if high - low < 1e-3:
        high = low + 1e-3  # guard: identical bounds make this blow up
    out = []
    for i in range(dim // 2):
        val = (i - low) / (high - low)
        out.append(min(1.0, max(0.0, val)))
    return out


def _yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    """Attention temperature correction. Without it, long context degrades even
    though the positions interpolate correctly."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@lru_cache(maxsize=32)
def inv_frequencies(head_dim: int, cfg: RopeConfig) -> tuple[float, ...]:
    """Per-dimension-pair inverse frequencies, after any scaling."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    base = [1.0 / (cfg.theta ** (i / head_dim)) for i in range(0, head_dim, 2)]

    if cfg.scaling is None or cfg.factor == 1.0:
        return tuple(base)

    if cfg.scaling == "linear":
        return tuple(f / cfg.factor for f in base)

    if cfg.scaling == "yarn":
        low = math.floor(
            _yarn_correction_dim(cfg.beta_fast, head_dim, cfg.theta, cfg.original_context)
        )
        high = math.ceil(
            _yarn_correction_dim(cfg.beta_slow, head_dim, cfg.theta, cfg.original_context)
        )
        ramp = _yarn_ramp(low, high, head_dim)
        # ramp == 1 -> keep the original (extrapolate), 0 -> fully interpolate
        return tuple(
            f * (r + (1.0 - r) / cfg.factor) for f, r in zip(base, ramp)
        )

    raise ValueError(f"unknown rope scaling {cfg.scaling!r}")


@lru_cache(maxsize=8)
def _cos_sin_table(head_dim: int, max_pos: int, cfg: RopeConfig) -> tuple[tuple, tuple]:
    freqs = inv_frequencies(head_dim, cfg)
    scale = _yarn_mscale(cfg.factor, cfg.mscale) if cfg.scaling == "yarn" else 1.0
    cos_t, sin_t = [], []
    for pos in range(max_pos):
        cos_t.append(tuple(math.cos(pos * f) * scale for f in freqs))
        sin_t.append(tuple(math.sin(pos * f) * scale for f in freqs))
    return tuple(cos_t), tuple(sin_t)


def rotate(vec: list[float], pos: int, cfg: RopeConfig | None = None) -> list[float]:
    """Apply RoPE to one head vector at absolute position `pos`.

    Uses the split-half convention (pair i with i + d/2), matching how the
    checkpoints store their projections. Pairing adjacent elements instead is
    also valid RoPE but is *not* interchangeable with these weights — it
    silently degrades quality rather than failing, which cost a week once.
    """
    cfg = cfg or RopeConfig()
    dim = len(vec)
    half = dim // 2
    freqs = inv_frequencies(dim, cfg)
    scale = _yarn_mscale(cfg.factor, cfg.mscale) if cfg.scaling == "yarn" else 1.0

    out = [0.0] * dim
    for i in range(half):
        angle = pos * freqs[i]
        cos_v, sin_v = math.cos(angle) * scale, math.sin(angle) * scale
        x1, x2 = vec[i], vec[i + half]
        out[i] = x1 * cos_v - x2 * sin_v
        out[i + half] = x2 * cos_v + x1 * sin_v
    return out


def rotate_batch(vectors: list[list[float]], positions: list[int], cfg: RopeConfig | None = None):
    if len(vectors) != len(positions):
        raise ValueError(f"{len(vectors)} vectors but {len(positions)} positions")
    return [rotate(v, p, cfg) for v, p in zip(vectors, positions)]


def effective_context(cfg: RopeConfig) -> int:
    """Positions the config can address before frequencies alias."""
    if cfg.scaling is None:
        return cfg.original_context
    return int(cfg.original_context * cfg.factor)
