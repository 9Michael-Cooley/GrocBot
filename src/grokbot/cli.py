"""Command line interface.

    grokbot chat    [--config ...] [--preset ...] [--persona ...]
    grokbot serve   [--host ...] [--port ...]
    grokbot agent   "prompt" [--preset ...] [--tools a,b]
    grokbot presets
    grokbot info    [--config ...]
    grokbot bench   [--requests N]
    grokbot tokens  "text"
    grokbot pet     [--species dog|cat]      (unreleased, needs GROKBOT_ENABLE_PETS=1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .errors import GrokBotError
from .utils.logging import configure, get_logger

log = get_logger(__name__)

DEFAULT_CONFIG = "configs/grok-3-mini.yaml"


def _resolve_config(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    # Allow running from anywhere in the tree, which is how everyone actually
    # invokes it.
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidate = parent / path
        if candidate.exists():
            return str(candidate)
    raise GrokBotError(f"config not found: {path}")


def _load(args) -> Config:
    cfg = Config.load(_resolve_config(args.config))
    overrides = {}
    if getattr(args, "weights", None):
        overrides["weights.path"] = args.weights
    if getattr(args, "max_running", None):
        overrides["scheduler.max_running"] = args.max_running
    if getattr(args, "kv_dtype", None):
        overrides["cache.dtype"] = args.kv_dtype
    if getattr(args, "seed", None) is not None:
        overrides["runtime.seed"] = args.seed
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_info(args) -> int:
    from .model.transformer import TransformerConfig

    cfg = _load(args)
    cfg.validate()
    model = TransformerConfig.from_config(cfg)
    print(model.summary())
    print()
    print(f"config             {cfg.source}")
    print(f"weights            {cfg.get('weights.path') or '(none - synthetic backend)'}")
    print(f"tensor parallel    {cfg.get('runtime.tensor_parallel', 1)}")
    print(f"block size         {cfg.get('cache.block_size', 16)}")
    print(f"max running        {cfg.get('scheduler.max_running', 256)}")
    return 0


def cmd_tokens(args) -> int:
    from .tokenizer import Tokenizer

    tok = Tokenizer.synthetic(vocab_size=8192)
    ids = tok.encode(args.text)
    print(f"text     {args.text!r}")
    print(f"tokens   {len(ids)}")
    print(f"ids      {ids}")
    print(f"pieces   {[tok.decode([i]) for i in ids]}")
    print(f"decoded  {tok.decode(ids)!r}")
    if tok.decode(ids) != args.text:
        print("WARNING: round-trip mismatch", file=sys.stderr)
        return 1
    return 0


def cmd_presets(args) -> int:
    from .agent.presets import describe_all

    print(describe_all(include_hidden=args.all))
    return 0


def cmd_chat(args) -> int:
    from .agent.persona import get as get_persona
    from .agent.presets import get_preset
    from .inference.engine import Engine

    cfg = _load(args)
    engine = Engine(cfg)

    preset = get_preset(args.preset)
    # --persona overrides the preset's persona; the preset still supplies sampling.
    persona = get_persona(args.persona) if args.persona else preset.get_persona()

    gen = preset.generation_config(
        max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed
    )
    print(f"preset: {preset.name} ({preset.description})")

    history: list[dict] = [{"role": "system", "content": persona.render()}]
    banner = f"grokbot {__version__} - {engine.model_config.name}"
    if engine.backend.is_synthetic:
        banner += "  [SYNTHETIC BACKEND: output is not model output]"
    print(banner)
    print("ctrl-c or /exit to quit, /reset to clear history, /stats for counters\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            history = history[:1]
            engine.reset()
            print("(history cleared)\n")
            continue
        if line == "/stats":
            print(json.dumps(engine.stats(), indent=2), "\n")
            continue

        history.append({"role": "user", "content": line})
        print("grok> ", end="", flush=True)
        pieces: list[str] = []
        try:
            for chunk in engine.stream_chat(history, gen):
                pieces.append(chunk.text)
                print(chunk.text, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print("\n")
        history.append({"role": "assistant", "content": "".join(pieces)})


def cmd_serve(args) -> int:
    from .serve.api import serve

    cfg = _load(args)
    serve(cfg, host=args.host, port=args.port)
    return 0


def cmd_agent(args) -> int:
    from .agent.presets import get_preset
    from .inference.engine import Engine
    from .tools.registry import REGISTRY

    cfg = _load(args)
    engine = Engine(cfg)
    preset = get_preset(args.preset)

    overrides: dict = {"max_iterations": args.max_iterations}
    if args.tools:
        overrides["tools"] = REGISTRY.subset([t.strip() for t in args.tools.split(",")])
    if args.persona:
        from .agent.persona import get as get_persona

        overrides["persona"] = get_persona(args.persona)

    agent = preset.build_agent(engine, **overrides)
    result = agent.run(args.prompt, preset.generation_config(seed=args.seed))
    for step in result.steps:
        print(f"--- step {step.iteration} ({step.duration_s * 1000:.0f} ms)")
        if step.text:
            print(step.text)
        for call, outcome in zip(step.tool_calls, step.results):
            status = "ok" if outcome["ok"] else "ERROR"
            detail = outcome.get("result", outcome.get("error", ""))
            print(f"  [{status}] {call.name}({json.dumps(call.arguments)}) -> {detail}")
    print(f"\n=== {result.stopped_because} "
          f"({result.iterations} iterations, {result.total_tool_calls} tool calls, "
          f"{result.duration_s:.2f}s)")
    print(result.answer)
    return 0


def cmd_pet(args) -> int:
    """Unreleased. See docs/upcoming.md."""
    from .experimental.pets import ENABLE_FLAG, Pet, PetBot, PetsDisabled, available_species

    try:
        pet = Pet.create(args.species, args.name)
    except PetsDisabled as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"\n  {ENABLE_FLAG}=1 python -m grokbot pet --species "
              f"{'|'.join(available_species())}", file=sys.stderr)
        return 1

    from .inference.engine import Engine

    engine = Engine(_load(args))
    bot = PetBot(engine, pet)

    print(f"{pet.status_line()}")
    print("/feed [food]  /play [activity]  /pet  /nap  /status  /exit\n")

    while True:
        try:
            line = input(f"{pet.name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if command in ("exit", "quit"):
                return 0
            if command == "feed":
                print(f"  {bot.feed(rest)}")
            elif command == "play":
                print(f"  {bot.play(rest or 'fetch')}")
            elif command == "pet":
                print(f"  {bot.pet_it()}")
            elif command == "nap":
                print(f"  {pet.nap(float(rest) if rest else 1.0)}")
            elif command == "status":
                print(f"  {pet.status_line()}")
            else:
                print(f"  unknown command /{command}")
            print(f"  [{pet.status_line()}]\n")
            continue

        print(f"{pet.name}: ", end="", flush=True)
        for chunk in bot.stream(line):
            print(chunk.text, end="", flush=True)
        print(f"\n  [{pet.status_line()}]\n")


def cmd_bench(args) -> int:
    from .inference.engine import Engine

    cfg = _load(args)
    engine = Engine(cfg)
    stats = engine.benchmark_step(args.requests, args.prompt_tokens, args.output_tokens)
    print(json.dumps(stats, indent=2))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grokbot", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"grokbot {__version__}")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--weights", default=None, help="path to a checkpoint directory")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--max-running", type=int, default=None, help="scheduler.max_running")
        p.add_argument("--kv-dtype", default=None, choices=["auto", "bf16", "fp16", "fp8"])
        return p

    p = common(sub.add_parser("info", help="print model and runtime configuration"))
    p.set_defaults(func=cmd_info)

    p = common(sub.add_parser("chat", help="interactive chat"))
    p.add_argument("--preset", default="default", help="bot preset (see `grokbot presets`)")
    p.add_argument("--persona", default=None, help="override the preset's persona")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("presets", help="list bot presets")
    p.add_argument("--all", action="store_true", help="include hidden presets")
    p.set_defaults(func=cmd_presets)

    p = common(sub.add_parser("pet", help="companion bot (UNRELEASED, see docs/upcoming.md)"))
    p.add_argument("--species", default="dog", choices=["dog", "cat"])
    p.add_argument("--name", default=None, help="defaults to Odie (dog) / Garfield (cat)")
    p.set_defaults(func=cmd_pet)

    p = common(sub.add_parser("serve", help="run the HTTP server"))
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = common(sub.add_parser("agent", help="run the agent loop once"))
    p.add_argument("prompt")
    p.add_argument("--preset", default="default", help="bot preset (see `grokbot presets`)")
    p.add_argument("--tools", default="", help="comma-separated tool names")
    p.add_argument("--persona", default=None, help="override the preset's persona")
    p.add_argument("--max-iterations", type=int, default=8)
    p.set_defaults(func=cmd_agent)

    p = common(sub.add_parser("bench", help="drive synthetic load through the scheduler"))
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=64)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("tokens", help="tokenize text and show the pieces")
    p.add_argument("text")
    p.set_defaults(func=cmd_tokens)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(args.log_level)
    try:
        return args.func(args)
    except GrokBotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
