#!/usr/bin/env python
"""Streaming chat with history.

    python examples/chat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokbot import Engine, GenerationConfig  # noqa: E402
from grokbot.agent.persona import get as get_persona  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grok-3-mini.yaml"


def main() -> int:
    engine = Engine.from_config(CONFIG)
    gen = GenerationConfig(temperature=0.7, top_p=0.95, max_tokens=512, seed=0)

    history = [{"role": "system", "content": get_persona("default").render()}]

    print(f"{engine.model_config.name} — ctrl-c to quit")
    if engine.backend.is_synthetic:
        print("(synthetic backend: output is not model output)")

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue

        history.append({"role": "user", "content": user})

        print("grok> ", end="", flush=True)
        pieces = []
        for chunk in engine.stream_chat(history, gen):
            pieces.append(chunk.text)
            print(chunk.text, end="", flush=True)
            if chunk.is_final:
                print(f"\n  [{chunk.finish_reason.value}]")

        history.append({"role": "assistant", "content": "".join(pieces)})

        # Keep the system turn plus the last 10 exchanges. Real deployments
        # should use agent.WorkingMemory, which evicts by priority rather than
        # by age.
        if len(history) > 21:
            history = history[:1] + history[-20:]


if __name__ == "__main__":
    raise SystemExit(main())
