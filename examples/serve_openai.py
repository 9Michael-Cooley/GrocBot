#!/usr/bin/env python
"""Start the server and drive it with plain HTTP, streaming and not.

    python examples/serve_openai.py

Uses urllib so it runs with no dependencies. Any OpenAI-compatible client
library works against the same endpoints.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokbot.config import Config  # noqa: E402
from grokbot.serve.api import serve  # noqa: E402
from grokbot.utils.logging import configure  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "serving.yaml"
HOST, PORT = "127.0.0.1", 8123
BASE = f"http://{HOST}:{PORT}"


def post(path: str, payload: dict):
    request = urllib.request.Request(
        BASE + path,
        json.dumps(payload).encode(),
        {"content-type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=60)


def main() -> int:
    configure("warning")
    cfg = Config.load(CONFIG)

    threading.Thread(target=serve, args=(cfg, HOST, PORT), daemon=True).start()
    for _ in range(50):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        print("server did not come up", file=sys.stderr)
        return 1

    print("models:", [m["id"] for m in json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"]])

    # --- non-streaming ---
    body = {
        "model": "grok-3-mini",
        "messages": [{"role": "user", "content": "explain paged attention"}],
        "max_tokens": 48,
        "temperature": 0.7,
    }
    result = json.load(post("/v1/chat/completions", body))
    print("\n--- completion ---")
    print(result["choices"][0]["message"]["content"])
    print("finish:", result["choices"][0]["finish_reason"])
    print("usage :", result["usage"])

    # --- streaming ---
    print("\n--- streaming ---")
    body["stream"] = True
    body["messages"] = [{"role": "user", "content": "how does the scheduler work?"}]
    with post("/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                print("\n[done]")
                break
            delta = json.loads(data)["choices"][0]["delta"]
            print(delta.get("content", ""), end="", flush=True)

    # --- error shape ---
    print("\n--- error handling ---")
    try:
        post("/v1/chat/completions", {"messages": []})
    except urllib.error.HTTPError as exc:
        print(exc.code, json.load(exc)["error"]["message"])

    print("\n--- stats ---")
    stats = json.load(urllib.request.urlopen(BASE + "/stats"))
    print(json.dumps(stats["engine"]["scheduler"], indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
