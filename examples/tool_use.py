#!/usr/bin/env python
"""Registering custom tools and running the agent loop.

    python examples/tool_use.py

Note the agent will not produce sensible tool calls under the synthetic backend —
it emits seeded noise, not decisions. This example is about the wiring: schema
generation, registry isolation, sandboxing, and the loop's guards.
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokbot import Engine  # noqa: E402
from grokbot.agent import Agent  # noqa: E402
from grokbot.tools import REGISTRY, Sandbox, SandboxPolicy  # noqa: E402
from grokbot.tools.registry import ToolRegistry, tool  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grok-3-mini.yaml"

# A private registry so these don't leak into the global one. Note the
# `registry=` keyword — ToolRegistry defines __len__, so an empty registry is
# falsy and `registry or REGISTRY` would quietly use the global one.
tools = ToolRegistry()

# Reuse a builtin.
tools.register(REGISTRY.get("calculator"))


@tool(description="Look up the current price of a stock ticker.", registry=tools)
def stock_price(symbol: str, currency: Literal["USD", "EUR", "GBP"] = "USD") -> float:
    """Fake market data — a real one would call a quote service."""
    table = {"NVDA": 118.42, "TSLA": 242.10, "AAPL": 227.55}
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79}
    if symbol.upper() not in table:
        raise ValueError(f"unknown ticker {symbol!r}")
    return round(table[symbol.upper()] * rates[currency], 2)


@tool(description="Search internal documentation. Returns matching excerpts.",
      registry=tools, timeout_s=5.0)
def doc_search(query: str, limit: int = 3) -> list[str]:
    corpus = {
        "kv cache": "Attention state is stored in fixed-size blocks of 16 tokens.",
        "preemption": "Preemption is LIFO and thrashes above ~400 concurrent (GROK-4417).",
        "moe": "Two of eight experts run per token; the router is fp32.",
    }
    hits = [v for k, v in corpus.items() if k in query.lower()]
    return hits[:limit] or ["no matches"]


def main() -> int:
    print("registered:", tools.names(), "\n")

    schema = tools.get("stock_price").schema()["function"]
    print("generated schema for stock_price:")
    print(f"  description: {schema['description']}")
    for name, spec in schema["parameters"]["properties"].items():
        required = name in schema["parameters"]["required"]
        print(f"  {name}: {spec} {'(required)' if required else ''}")

    print("\ndirect calls:")
    print("  stock_price(NVDA)      =", tools.get("stock_price")(symbol="NVDA"))
    print("  stock_price(NVDA, EUR) =", tools.get("stock_price")(symbol="NVDA", currency="EUR"))
    print("  calculator(17*23)      =", tools.get("calculator")(expression="17*23"))
    print("  doc_search('moe')      =", tools.get("doc_search")(query="moe"))

    print("\nvalidation:")
    for bad_args in ({"symbol": "NVDA", "currency": "JPY"}, {}, {"symbol": "ZZZZ"}):
        try:
            print("  ", bad_args, "->", tools.get("stock_price")(**bad_args))
        except Exception as exc:
            print("  ", bad_args, "->", f"{type(exc).__name__}: {exc}")

    print("\nsandboxed execution:")
    sandbox = Sandbox(SandboxPolicy(wall_clock_s=5, memory_mb=256))
    result = sandbox.run(tools.get("calculator").fn, {"expression": "sqrt(144) * 3"})
    print(f"   value={result.value} in {result.duration_s * 1000:.1f}ms "
          f"(process-isolated: {result.isolated})")

    print("\nagent loop:")
    engine = Engine.from_config(CONFIG)
    agent = Agent(engine, tools=tools, max_iterations=3)
    outcome = agent.run("What is NVDA trading at, and what is that times 100?")

    for step in outcome.steps:
        print(f"  step {step.iteration}: {len(step.tool_calls)} call(s), "
              f"{step.duration_s * 1000:.0f}ms")
        for call, res in zip(step.tool_calls, step.results):
            status = "ok" if res["ok"] else "ERROR"
            print(f"    [{status}] {call.name}({call.arguments}) -> "
                  f"{res.get('result', res.get('error'))}")
    print(f"  stopped: {outcome.stopped_because}")
    print(f"  answer:  {outcome.answer[:120]}")
    print(f"  memory:  {agent.memory.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
