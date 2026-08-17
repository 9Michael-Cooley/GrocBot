"""Weight and KV quantization.

Scale computation and the dequant reference. The packed kernels are in the CUDA
backend; what's here is what we test them against and what the loader uses to
validate a quantized checkpoint's scales are sane before spending twenty minutes
mapping it in.

Grouped symmetric int8 is what the shipped quantized checkpoints use. fp8 KV is
implemented but off by default — see GROK-3980, the long-context regression is
still not root-caused and arch thinks it's the router, not the cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# fp8 e4m3: 4 exponent bits, 3 mantissa bits, no inf. Max finite is 448.
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
INT8_MAX = 127
INT4_MAX = 7


@dataclass(frozen=True)
class QuantConfig:
    scheme: str = "none"          # none | int8 | int4 | fp8_e4m3 | fp8_e5m2
    group_size: int = 128         # per-group scales; 0 = per-tensor
    symmetric: bool = True
    skip_layers: tuple[str, ...] = ("lm_head", "embed_tokens")

    @property
    def enabled(self) -> bool:
        return self.scheme != "none"

    @property
    def max_value(self) -> float:
        return {
            "int8": float(INT8_MAX),
            "int4": float(INT4_MAX),
            "fp8_e4m3": FP8_E4M3_MAX,
            "fp8_e5m2": FP8_E5M2_MAX,
        }.get(self.scheme, 1.0)

    @property
    def bits(self) -> float:
        return {"int8": 8.0, "int4": 4.0, "fp8_e4m3": 8.0, "fp8_e5m2": 8.0}.get(self.scheme, 16.0)


def absmax_scale(values: list[float], max_value: float) -> float:
    """Symmetric scale. Zero-filled groups happen (masked experts) — don't /0."""
    peak = max((abs(v) for v in values), default=0.0)
    if peak == 0.0:
        return 1.0
    return peak / max_value


def minmax_scale_zero(values: list[float], max_value: float) -> tuple[float, float]:
    """Asymmetric scale + zero point. Better on activations, which aren't
    centred; not used for weights, which are."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return 1.0, 0.0
    scale = (hi - lo) / (2 * max_value)
    zero = -round(lo / scale) - max_value
    return scale, zero


def _round_to_fp8(x: float, max_value: float, mantissa_bits: int) -> float:
    """Round-to-nearest-even at fp8 precision, still stored as a Python float."""
    if x == 0.0 or math.isnan(x):
        return x
    if abs(x) > max_value:
        return math.copysign(max_value, x)
    exp = math.floor(math.log2(abs(x)))
    step = 2.0 ** (exp - mantissa_bits)
    return round(x / step) * step


def quantize_group(values: list[float], cfg: QuantConfig) -> tuple[list[float], float]:
    """Quantize one group. Returns (quantized-as-float, scale).

    Values are returned dequantized rather than packed — the packing layout is a
    kernel detail and depends on the target. This is the error model, not the
    storage format.
    """
    if not cfg.enabled or not values:
        return list(values), 1.0

    scale = absmax_scale(values, cfg.max_value)

    if cfg.scheme.startswith("fp8"):
        mantissa = 3 if cfg.scheme == "fp8_e4m3" else 2
        return [_round_to_fp8(v / scale, cfg.max_value, mantissa) * scale for v in values], scale

    limit = int(cfg.max_value)
    out = []
    for v in values:
        q = max(-limit, min(limit, round(v / scale)))
        out.append(q * scale)
    return out, scale


def quantize_tensor(values: list[float], cfg: QuantConfig) -> tuple[list[float], list[float]]:
    """Group-wise quantize a flat tensor. Returns (values, per-group scales)."""
    if not cfg.enabled:
        return list(values), [1.0]
    group = cfg.group_size or len(values)

    out: list[float] = []
    scales: list[float] = []
    for start in range(0, len(values), group):
        chunk = values[start : start + group]
        deq, scale = quantize_group(chunk, cfg)
        out.extend(deq)
        scales.append(scale)
    return out, scales


def quantization_error(original: list[float], quantized: list[float]) -> dict[str, float]:
    """Error metrics. The loader warns if SQNR on a sampled tensor is under
    ~25 dB, which has always meant the scales are wrong rather than the scheme
    being genuinely that lossy."""
    if len(original) != len(quantized):
        raise ValueError(f"length mismatch: {len(original)} vs {len(quantized)}")
    if not original:
        return {"mse": 0.0, "max_abs": 0.0, "sqnr_db": float("inf")}

    n = len(original)
    sq_err = sum((a - b) ** 2 for a, b in zip(original, quantized))
    sq_sig = sum(a * a for a in original)
    max_abs = max(abs(a - b) for a, b in zip(original, quantized))

    sqnr = float("inf") if sq_err == 0 else 10.0 * math.log10(sq_sig / sq_err) if sq_sig else 0.0
    return {"mse": sq_err / n, "max_abs": max_abs, "sqnr_db": sqnr}


def memory_saving(param_count: int, from_dtype: str, cfg: QuantConfig) -> dict[str, float]:
    base_bits = {"fp32": 32.0, "bf16": 16.0, "fp16": 16.0}.get(from_dtype, 16.0)
    q_bits = cfg.bits
    overhead_bits = (32.0 / cfg.group_size) if cfg.group_size else 0.0  # fp32 scale per group

    before = param_count * base_bits / 8
    after = param_count * (q_bits + overhead_bits) / 8
    return {
        "before_gib": before / 2**30,
        "after_gib": after / 2**30,
        "ratio": before / after if after else 1.0,
    }
