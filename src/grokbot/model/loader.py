"""Checkpoint loading.

Reads safetensors headers and validates the tensor manifest against the shapes
TransformerConfig expects, so a mismatched checkpoint fails in seconds with a
useful message instead of twenty minutes in with an OOM or silent garbage.

Only headers are parsed here. Actually mapping tensor data is the backend's job
(it needs to land in device memory, not a Python list). If no checkpoint is
configured, the loader says so loudly and returns a SyntheticBackend.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ConfigError, WeightsNotFound
from ..tokenizer import Tokenizer
from ..utils.logging import get_logger
from .backend import Backend, create_backend
from .transformer import TransformerConfig

log = get_logger(__name__)

_SAFETENSORS_MAX_HEADER = 100_000_000   # sanity bound; real headers are ~1 MB


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.offsets[1] - self.offsets[0]


def read_safetensors_header(path: Path) -> dict[str, TensorInfo]:
    """Parse a safetensors header: u64 LE length, then that many bytes of JSON."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ConfigError(f"{path.name} is truncated (under 8 bytes)")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > _SAFETENSORS_MAX_HEADER:
            raise ConfigError(f"{path.name}: implausible header length {header_len}")
        blob = fh.read(header_len)
        if len(blob) < header_len:
            raise ConfigError(f"{path.name}: header truncated")

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name}: header is not valid JSON: {exc}") from exc

    tensors: dict[str, TensorInfo] = {}
    for name, meta in parsed.items():
        if name == "__metadata__":
            continue
        try:
            tensors[name] = TensorInfo(
                name=name,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                offsets=tuple(meta["data_offsets"]),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{path.name}: bad entry for tensor {name!r}: {exc}") from exc
    return tensors


def expected_tensors(cfg: TransformerConfig) -> dict[str, tuple[int, ...]]:
    """The manifest a checkpoint must satisfy. Names follow the HF-style layout
    the conversion script emits (scripts/convert_checkpoint.py)."""
    h, hd = cfg.hidden_size, cfg.head_dim
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (cfg.vocab_size, h),
        "model.norm.weight": (h,),
    }
    if not cfg.tie_word_embeddings:
        expected["lm_head.weight"] = (cfg.vocab_size, h)

    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        expected[f"{p}.self_attn.q_proj.weight"] = (cfg.num_heads * hd, h)
        expected[f"{p}.self_attn.k_proj.weight"] = (cfg.num_kv_heads * hd, h)
        expected[f"{p}.self_attn.v_proj.weight"] = (cfg.num_kv_heads * hd, h)
        expected[f"{p}.self_attn.o_proj.weight"] = (h, cfg.num_heads * hd)
        expected[f"{p}.input_layernorm.weight"] = (h,)
        expected[f"{p}.post_attention_layernorm.weight"] = (h,)

        if cfg.is_moe:
            moe = cfg.moe
            assert moe is not None
            expected[f"{p}.mlp.gate.weight"] = (moe.num_experts, h)
            for e in range(moe.num_experts):
                ep = f"{p}.mlp.experts.{e}"
                expected[f"{ep}.gate_proj.weight"] = (moe.expert_intermediate_size, h)
                expected[f"{ep}.up_proj.weight"] = (moe.expert_intermediate_size, h)
                expected[f"{ep}.down_proj.weight"] = (h, moe.expert_intermediate_size)
            if moe.shared_expert:
                sp = f"{p}.mlp.shared_expert"
                expected[f"{sp}.gate_proj.weight"] = (cfg.intermediate_size, h)
                expected[f"{sp}.up_proj.weight"] = (cfg.intermediate_size, h)
                expected[f"{sp}.down_proj.weight"] = (h, cfg.intermediate_size)
        else:
            expected[f"{p}.mlp.gate_proj.weight"] = (cfg.intermediate_size, h)
            expected[f"{p}.mlp.up_proj.weight"] = (cfg.intermediate_size, h)
            expected[f"{p}.mlp.down_proj.weight"] = (h, cfg.intermediate_size)
    return expected


def validate_manifest(
    found: dict[str, TensorInfo], cfg: TransformerConfig, *, strict: bool = True
) -> list[str]:
    """Compare a checkpoint's tensors against the config. Returns problems."""
    expected = expected_tensors(cfg)
    problems: list[str] = []

    missing = sorted(set(expected) - set(found))
    if missing:
        head = ", ".join(missing[:5])
        problems.append(
            f"{len(missing)} tensor(s) missing from checkpoint: {head}"
            + (" ..." if len(missing) > 5 else "")
        )

    extra = sorted(set(found) - set(expected))
    if extra and strict:
        head = ", ".join(extra[:5])
        problems.append(
            f"{len(extra)} unexpected tensor(s): {head}" + (" ..." if len(extra) > 5 else "")
        )

    for name, want in expected.items():
        info = found.get(name)
        if info is not None and tuple(info.shape) != want:
            problems.append(f"{name}: checkpoint has {tuple(info.shape)}, config wants {want}")
            if len(problems) > 20:
                problems.append("... further shape mismatches suppressed")
                break
    return problems


def discover_shards(root: Path) -> list[Path]:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise WeightsNotFound(f"no .safetensors shards under {root}")
    return shards


def load_backend(cfg: Config, *, strict: bool = False) -> tuple[Backend, TransformerConfig, Tokenizer]:
    """Build a backend from config. Falls back to synthetic when no weights."""
    model_cfg = TransformerConfig.from_config(cfg)

    weights_path = cfg.get("weights.path")
    if not weights_path:
        if strict:
            raise WeightsNotFound(
                "weights.path is unset. Set it in the config, pass --weights, or export "
                "GROKBOT_WEIGHTS."
            )
        log.warning(
            "no weights configured for %s; falling back to the synthetic backend",
            model_cfg.name,
        )
        tokenizer = _load_tokenizer(cfg, model_cfg)
        return (
            create_backend("synthetic", model_cfg, tokenizer, seed=cfg.get("runtime.seed", 0)),
            model_cfg,
            tokenizer,
        )

    root = Path(weights_path)
    if not root.exists():
        raise WeightsNotFound(f"weights.path does not exist: {root}")

    shards = discover_shards(root)
    manifest: dict[str, TensorInfo] = {}
    for shard in shards:
        for name, info in read_safetensors_header(shard).items():
            if name in manifest:
                log.warning("tensor %s appears in more than one shard; last wins", name)
            manifest[name] = info

    problems = validate_manifest(manifest, model_cfg, strict=strict)
    if problems:
        joined = "\n  ".join(problems)
        if strict:
            raise ConfigError(f"checkpoint does not match {model_cfg.name}:\n  {joined}")
        log.warning("checkpoint/config mismatches (continuing):\n  %s", joined)

    total_params = sum(t.num_elements for t in manifest.values())
    expected_params = model_cfg.param_count()["total"]
    drift = abs(total_params - expected_params) / expected_params if expected_params else 0
    log.info(
        "loaded manifest: %d tensors, %.1fB params across %d shard(s) (config expects %.1fB, %.1f%% drift)",
        len(manifest),
        total_params / 1e9,
        len(shards),
        expected_params / 1e9,
        drift * 100,
    )

    tokenizer = _load_tokenizer(cfg, model_cfg)
    backend_name = cfg.get("runtime.backend", "cuda")
    return create_backend(backend_name, model_cfg, tokenizer), model_cfg, tokenizer


def _load_tokenizer(cfg: Config, model_cfg: TransformerConfig) -> Tokenizer:
    explicit = cfg.get("weights.tokenizer")
    weights = cfg.get("weights.path")

    candidate = Path(explicit) if explicit else (Path(weights) / "tokenizer.json" if weights else None)
    if candidate and candidate.exists():
        log.info("tokenizer: %s", candidate)
        return Tokenizer.from_file(candidate)

    if candidate:
        log.warning("tokenizer not found at %s; using synthetic vocab", candidate)
    # Synthetic vocab is deliberately much smaller than the real one — building
    # 131k merges per process start costs seconds and buys nothing here.
    return Tokenizer.synthetic(vocab_size=8192, seed=cfg.get("runtime.seed", 0))
