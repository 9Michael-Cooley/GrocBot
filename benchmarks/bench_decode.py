#!/usr/bin/env python
"""Decode-throughput harness.

    python benchmarks/bench_decode.py --config configs/grok-3-mini.yaml --requests 128

Drives synthetic load through the real scheduler, cache, and sampler. Without
the CUDA backend the *compute* is fake, so tok/s here measures Python overhead
in the serving path, not model throughput. It is still the right harness for
scheduler work: preemption counts, cache hit rates, and batch occupancy are all
real.

The numbers in results.md came from this harness against real weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokbot.config import Config                       # noqa: E402
from grokbot.inference.engine import Engine             # noqa: E402
from grokbot.inference.sampler import GenerationConfig  # noqa: E402
from grokbot.inference.scheduler import Request         # noqa: E402
from grokbot.inference.stream import StreamAssembler    # noqa: E402
from grokbot.utils.logging import configure             # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def run(engine: Engine, requests: int, prompt_tokens: int, output_tokens: int, unique: bool):
    gen = GenerationConfig(max_tokens=output_tokens, temperature=0.0)
    tracked: list[Request] = []

    for i in range(requests):
        # Distinct prompts unless asked otherwise: identical ones share every
        # block through the prefix cache and the run stops measuring anything.
        seed_text = f"benchmark request {i} " if unique else "benchmark request "
        ids = engine.tokenizer.encode(seed_text * max(1, prompt_tokens // 4))[:prompt_tokens]
        req = Request(prompt_token_ids=ids or [1], gen_config=gen)
        engine.scheduler.add_request(req)
        engine._assemblers[req.request_id] = StreamAssembler(engine.tokenizer, gen)
        tracked.append(req)

    start = time.monotonic()
    steps = 0
    while engine.scheduler.has_work:
        engine._step()
        steps += 1
        if steps > requests * output_tokens * 4:
            raise SystemExit("scheduler did not converge — likely thrashing (GROK-4417)")
    elapsed = time.monotonic() - start

    ttfts = [r.ttft() for r in tracked if r.ttft() is not None]
    latencies = [r.latency() for r in tracked if r.latency() is not None]
    snap = engine.scheduler.snapshot()

    return {
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": round(elapsed, 4),
        "steps": steps,
        "decode_tokens": snap["decode_tokens"],
        "prefill_tokens": snap["prefill_tokens"],
        "decode_tok_s": round(snap["decode_tokens"] / elapsed, 1) if elapsed else 0.0,
        "prefill_tok_s": round(snap["prefill_tokens"] / elapsed, 1) if elapsed else 0.0,
        "ttft_p50_ms": round(statistics.median(ttfts) * 1000, 2) if ttfts else 0.0,
        "ttft_p99_ms": round(percentile(ttfts, 0.99) * 1000, 2) if ttfts else 0.0,
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 2) if latencies else 0.0,
        "latency_p99_ms": round(percentile(latencies, 0.99) * 1000, 2) if latencies else 0.0,
        "preemptions": snap["preemptions"],
        "avg_batch": round(snap["decode_tokens"] / steps, 2) if steps else 0.0,
        "cache": snap["cache"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/grok-3-mini.yaml")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--max-running", type=int, default=None)
    ap.add_argument("--shared-prefix", action="store_true",
                    help="reuse one prompt so prefix caching engages")
    ap.add_argument("--sweep", action="store_true", help="sweep concurrency 1..128")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    configure("warning")

    cfg = Config.load(args.config)
    if args.max_running:
        cfg.set("scheduler.max_running", args.max_running)

    if args.sweep:
        results = []
        for concurrency in (1, 4, 16, 64, 128):
            cfg.set("scheduler.max_running", concurrency)
            engine = Engine(cfg)
            results.append(
                {"concurrency": concurrency,
                 **run(engine, concurrency, args.prompt_tokens, args.output_tokens,
                       not args.shared_prefix)}
            )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"{'conc':>5} {'decode tok/s':>13} {'ttft p50':>10} {'ttft p99':>10} {'preempt':>8}")
            for r in results:
                print(f"{r['concurrency']:>5} {r['decode_tok_s']:>13.1f} "
                      f"{r['ttft_p50_ms']:>9.1f}m {r['ttft_p99_ms']:>9.1f}m {r['preemptions']:>8}")
        return 0

    engine = Engine(cfg)
    if engine.backend.is_synthetic:
        print("NOTE: synthetic backend — this measures serving-path overhead, "
              "not model throughput.\n", file=sys.stderr)

    result = run(engine, args.requests, args.prompt_tokens, args.output_tokens,
                 not args.shared_prefix)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key in ("requests", "elapsed_s", "steps", "avg_batch", "decode_tok_s",
                    "prefill_tok_s", "ttft_p50_ms", "ttft_p99_ms", "latency_p50_ms",
                    "latency_p99_ms", "preemptions"):
            print(f"{key:>16}  {result[key]}")
        c = result["cache"]
        print(f"{'cache hits':>16}  {c['cache_hits']} / {c['cache_hits'] + c['cache_misses']}")
        print(f"{'evictions':>16}  {c['evictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
