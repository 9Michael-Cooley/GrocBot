#!/usr/bin/env python
"""Convert a training checkpoint to the serving layout.

    python scripts/convert_checkpoint.py --src /ckpt/step_412000 --dst /weights/grok-3

Training writes sharded optimizer state in the trainer's own layout. Serving
wants safetensors with the tensor names in model/loader.py::expected_tensors.

The tensor *rewriting* half needs the trainer's reader, which is in
//research/training and not here. What works standalone is the validation half:
point it at an already-converted directory with --check and it verifies the
manifest against a config before you spend twenty minutes loading it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokbot.config import Config  # noqa: E402
from grokbot.errors import GrokBotError  # noqa: E402
from grokbot.model.loader import (  # noqa: E402
    discover_shards,
    expected_tensors,
    read_safetensors_header,
    validate_manifest,
)
from grokbot.model.transformer import TransformerConfig  # noqa: E402
from grokbot.utils.logging import configure, get_logger  # noqa: E402

log = get_logger("convert")

# trainer name -> serving name. The trainer keeps QKV fused; serving wants them
# split, which is the one transformation that isn't a rename.
NAME_MAP = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "layers.{i}.attention_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "layers.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "layers.{i}.attention.wqkv.weight": "<SPLIT: q_proj, k_proj, v_proj>",
    "layers.{i}.attention.wo.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "layers.{i}.feed_forward.router.weight": "model.layers.{i}.mlp.gate.weight",
    "layers.{i}.feed_forward.experts.{e}.w1.weight": "model.layers.{i}.mlp.experts.{e}.gate_proj.weight",
    "layers.{i}.feed_forward.experts.{e}.w3.weight": "model.layers.{i}.mlp.experts.{e}.up_proj.weight",
    "layers.{i}.feed_forward.experts.{e}.w2.weight": "model.layers.{i}.mlp.experts.{e}.down_proj.weight",
}


def cmd_check(args) -> int:
    cfg = Config.load(args.config)
    cfg.validate()
    model = TransformerConfig.from_config(cfg)

    root = Path(args.check)
    shards = discover_shards(root)
    log.info("found %d shard(s) under %s", len(shards), root)

    manifest = {}
    for shard in shards:
        for name, info in read_safetensors_header(shard).items():
            if name in manifest:
                log.warning("%s appears in more than one shard", name)
            manifest[name] = info

    problems = validate_manifest(manifest, model, strict=args.strict)

    total = sum(t.num_elements for t in manifest.values())
    want = model.param_count()["total"]
    print(f"model            {model.name}")
    print(f"tensors          {len(manifest)} found / {len(expected_tensors(model))} expected")
    print(f"parameters       {total / 1e9:.2f}B found / {want / 1e9:.2f}B expected")
    print(f"drift            {abs(total - want) / want * 100:.2f}%")

    tokenizer = root / "tokenizer.json"
    print(f"tokenizer.json   {'present' if tokenizer.exists() else 'MISSING'}")
    if not tokenizer.exists():
        problems.append("tokenizer.json is missing; the loader will fall back to a synthetic vocab")

    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK")
    return 0


def cmd_manifest(args) -> int:
    cfg = Config.load(args.config)
    model = TransformerConfig.from_config(cfg)
    expected = expected_tensors(model)
    if args.json:
        print(json.dumps({k: list(v) for k, v in expected.items()}, indent=2))
    else:
        for name, shape in expected.items():
            print(f"{name:<64} {shape}")
        print(f"\n{len(expected)} tensors, {model.param_count()['total'] / 1e9:.2f}B parameters")
    return 0


def cmd_convert(args) -> int:
    raise GrokBotError(
        "conversion requires the trainer checkpoint reader from //research/training, "
        "which is not part of this tree. Use --check to validate an already-converted "
        "directory, or --manifest to print the expected tensor layout."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/grok-3-mini.yaml")
    ap.add_argument("--src", help="trainer checkpoint directory")
    ap.add_argument("--dst", help="output directory")
    ap.add_argument("--check", help="validate an existing serving checkpoint")
    ap.add_argument("--manifest", action="store_true", help="print expected tensors")
    ap.add_argument("--strict", action="store_true", help="treat extra tensors as errors")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    configure("info")
    try:
        if args.check:
            return cmd_check(args)
        if args.manifest:
            return cmd_manifest(args)
        if args.src:
            return cmd_convert(args)
        ap.print_help()
        return 2
    except GrokBotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
