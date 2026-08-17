"""Model definition and parameter accounting.

No tensors here — this is the shape/size description that the loader validates a
checkpoint against and that the memory planner sizes the KV cache from. Getting
these counts wrong shows up as an OOM twenty minutes into a load, so the
arithmetic is spelled out rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..errors import ConfigError
from .attention import AttentionConfig
from .moe import MoEConfig
from .rope import RopeConfig, effective_context

_DTYPE_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


@dataclass
class TransformerConfig:
    name: str = "grok-3-mini"
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 131072
    max_position_embeddings: int = 131072

    norm: str = "rmsnorm"
    norm_eps: float = 1e-5
    activation: str = "silu"
    tie_word_embeddings: bool = False
    dtype: str = "bf16"

    rope: RopeConfig = None          # type: ignore[assignment]
    moe: MoEConfig | None = None

    def __post_init__(self) -> None:
        if self.rope is None:
            self.rope = RopeConfig()
        if self.head_dim * self.num_heads != self.hidden_size:
            # Not fatal — grok-3 decouples head_dim from hidden_size on purpose —
            # but it's the first thing to check when a load produces garbage.
            pass

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> TransformerConfig:
        m = cfg.section("model")
        try:
            return cls(
                name=m.get("name", "unknown"),
                hidden_size=m.require("hidden_size"),
                intermediate_size=m.require("intermediate_size"),
                num_layers=m.require("num_layers"),
                num_heads=m.require("num_heads"),
                num_kv_heads=m.get("num_kv_heads") or m.require("num_heads"),
                head_dim=m.get("head_dim") or m.require("hidden_size") // m.require("num_heads"),
                vocab_size=m.require("vocab_size"),
                max_position_embeddings=m.get("max_position_embeddings", 8192),
                norm=m.get("norm", "rmsnorm"),
                norm_eps=m.get("norm_eps", 1e-5),
                activation=m.get("activation", "silu"),
                tie_word_embeddings=m.get("tie_word_embeddings", False),
                dtype=m.get("dtype", "bf16"),
                rope=RopeConfig.from_dict(m.get("rope")),
                moe=MoEConfig.from_dict(m.get("moe")),
            )
        except ConfigError as exc:
            raise ConfigError(f"invalid model section in {cfg.source}: {exc}") from exc

    @property
    def attention(self) -> AttentionConfig:
        return AttentionConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    # -- parameter accounting ---------------------------------------------

    def _attn_params(self) -> int:
        return self.attention.param_count()

    def _dense_ffn_params(self, intermediate: int) -> int:
        """SwiGLU: gate, up, down. Three matrices, not two."""
        gates = 2 if self.activation in ("silu", "swiglu", "geglu") else 1
        return (gates + 1) * self.hidden_size * intermediate

    def _ffn_params_per_layer(self) -> tuple[int, int]:
        """(total, active) FFN params for one layer."""
        if not self.is_moe:
            p = self._dense_ffn_params(self.intermediate_size)
            return p, p

        moe = self.moe
        assert moe is not None
        per_expert = self._dense_ffn_params(moe.expert_intermediate_size)
        router = self.hidden_size * moe.num_experts

        total = per_expert * moe.num_experts + router
        active = per_expert * moe.experts_per_token + router

        if moe.shared_expert:
            shared = self._dense_ffn_params(self.intermediate_size)
            total += shared
            active += shared          # shared expert runs for every token
        return total, active

    def _norm_params_per_layer(self) -> int:
        # RMSNorm has scale only, no bias. Two per block (pre-attn, pre-ffn).
        return 2 * self.hidden_size

    def _embedding_params(self) -> int:
        emb = self.vocab_size * self.hidden_size
        return emb if self.tie_word_embeddings else emb * 2

    def param_count(self) -> dict[str, int]:
        attn = self._attn_params()
        ffn_total, ffn_active = self._ffn_params_per_layer()
        norms = self._norm_params_per_layer()

        per_layer_total = attn + ffn_total + norms
        per_layer_active = attn + ffn_active + norms

        body_total = per_layer_total * self.num_layers
        body_active = per_layer_active * self.num_layers
        embeddings = self._embedding_params()
        final_norm = self.hidden_size

        return {
            "attention_per_layer": attn,
            "ffn_per_layer": ffn_total,
            "per_layer": per_layer_total,
            "embeddings": embeddings,
            "total": body_total + embeddings + final_norm,
            "active": body_active + embeddings + final_norm,
        }

    # -- memory ------------------------------------------------------------

    def weight_bytes(self, dtype: str | None = None) -> int:
        width = _DTYPE_BYTES.get(dtype or self.dtype, 2)
        return int(self.param_count()["total"] * width)

    def kv_bytes_per_token(self, dtype: str | None = None) -> int:
        width = _DTYPE_BYTES.get(dtype or self.dtype, 2)
        return int(2 * self.num_layers * self.num_kv_heads * self.head_dim * width)

    def plan_kv_blocks(
        self,
        total_memory_bytes: int,
        *,
        block_size: int = 16,
        utilization: float = 0.9,
        kv_dtype: str | None = None,
        tensor_parallel: int = 1,
    ) -> int:
        """How many cache blocks fit after weights and activation slack.

        Deliberately conservative — the activation reserve is a flat fraction
        rather than a real estimate, because the real number depends on the batch
        composition we haven't seen yet. Overshooting here OOMs mid-serve.
        """
        budget = total_memory_bytes * utilization
        weights = self.weight_bytes() / tensor_parallel
        activation_reserve = budget * 0.08
        free = budget - weights - activation_reserve
        if free <= 0:
            raise ConfigError(
                f"{self.name} weights ({weights / 2**30:.1f} GiB across tp={tensor_parallel}) "
                f"exceed the {budget / 2**30:.1f} GiB budget; raise tensor_parallel"
            )
        per_block = self.kv_bytes_per_token(kv_dtype) * block_size / tensor_parallel
        return int(free // per_block)

    def max_context(self) -> int:
        return min(self.max_position_embeddings, effective_context(self.rope))

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        p = self.param_count()
        b = 1_000_000_000
        lines = [
            f"model              {self.name}",
            f"arch               {'MoE' if self.is_moe else 'dense'} transformer, {self.num_layers} layers",
            f"hidden / ffn       {self.hidden_size} / {self.intermediate_size}",
            f"heads              {self.num_heads} q, {self.num_kv_heads} kv "
            f"(GQA {self.attention.group_size}:1), head_dim {self.head_dim}",
            f"vocab              {self.vocab_size}",
            f"context            {self.max_context():,} "
            f"(rope {self.rope.scaling or 'none'}, theta {self.rope.theta:g})",
        ]
        if self.is_moe:
            moe = self.moe
            assert moe is not None
            lines.append(
                f"experts            {moe.num_experts} total, {moe.experts_per_token} per token, "
                f"ffn {moe.expert_intermediate_size}"
                + (", + shared" if moe.shared_expert else "")
            )
        lines += [
            f"params total       {p['total'] / b:.1f}B",
            f"params active      {p['active'] / b:.1f}B",
            f"weights ({self.dtype})     {self.weight_bytes() / 2**30:.1f} GiB",
            f"kv per token       {self.kv_bytes_per_token() / 1024:.1f} KiB",
        ]
        return "\n".join(lines)
