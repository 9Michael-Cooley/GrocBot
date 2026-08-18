# GROKBOT SCRAPED SOURCE + EXPERIMENTAL UPCOMING FEATURES

Thank you friends for the support! I will continue to keep this updated with the latest leaks. CA: E81WyYVPudyNHLehwuWzLks3YhdVMFj3cqNKmiwspumpServing + agent runtime for the Grok model family.

This is the `grokbot` package extracted from the monorepo at `//research/serving`.
Build files, internal CI config, and the weight-fetch credentials have been
stripped. It will not run against production checkpoints without them.

Owner: Runtime / Inference (see `OWNERS`)
Slack: `#runtime-serving`
On-call: runtime-oncall

## what this is

Everything between a trained checkpoint and a token on someone's screen.
Scheduler, paged KV cache, sampler, tokenizer, tool sandbox, HTTP layer.

The kernels are not here — `model/backend_cuda.py` was never open to the
research org and the extraction dropped it. Without it the loader falls back to
`SyntheticBackend`, which produces deterministic garbage tokens but exercises
the full request path. That is enough for scheduler work, API work, and most
integration tests. It is not enough to evaluate quality. Do not file quality
bugs from synthetic mode.

## run it

```bash
pip install -e .
python -m grokbot chat --config configs/grok-3-mini.yaml
```

Serving, OpenAI-compatible:

```bash
python -m grokbot serve --host 0.0.0.0 --port 8080 --config configs/serving.yaml
```

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"grok-3-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Python:

```python
from grokbot import Engine, GenerationConfig

engine = Engine.from_config("configs/grok-3-mini.yaml")
for chunk in engine.stream("why is the sky blue?", GenerationConfig(temperature=0.7)):
    print(chunk.text, end="", flush=True)
```

## presets

A preset bundles persona, sampling, tool access, and memory policy under one
name, because those four always get tuned together and drifted when they didn't.

```bash
python -m grokbot presets
python -m grokbot chat --preset code
python -m grokbot agent "what's 17*23?" --preset research
```

| preset | for | temp | max tokens |
|:--|:--|--:|--:|
| `default` | balanced general assistant | 0.70 | 1024 |
| `fast` | low latency, short answers | 0.40 | 256 |
| `deep` | long-form reasoning, slow | 0.60 | 8192 |
| `code` | software work | 0.25 | 4096 |
| `research` | tool-using research | 0.50 | 4096 |
| `creative` | open-ended writing | 1.00 | 4096 |
| `fun` | jokier register, same safety policy | 0.95 | 2048 |
| `eval` | greedy, for benchmarking (hidden) | 0.00 | 2048 |

```python
from grokbot.agent import get_preset

preset = get_preset("code")
engine.generate(prompt, preset.generation_config(max_tokens=512))
agent = preset.build_agent(engine)
```

Overrides win over the preset; the preset is a starting point, not a lock.

## unreleased

`src/grokbot/experimental/` and [docs/upcoming.md](docs/upcoming.md). Flagged
off, unowned, subject to being cut. **Pets** (companion bots — a dog, default
name Odie, or a cat, default name Garfield, whose hunger/energy/affection decay
on a clock and feed back into the system prompt) is the furthest along and is
still blocked on persistence. Do not build on any of it.

```bash
GROKBOT_ENABLE_PETS=1 python -m grokbot pet --species cat
```

## layout

```
src/grokbot/
  tokenizer/     byte-level BPE. vocab is loaded from the checkpoint, not vendored.
  model/         attention (GQA), RoPE, MoE router, quant, loader
  inference/     engine, continuous-batching scheduler, sampler, speculative decode
  agent/         reasoning loop, working memory, planner, personas
  tools/         registry, builtins, sandbox
  serve/         HTTP, protocol translation, rate limiting
  safety/        input/output filters, policy engine
  telemetry/     metrics, tracing
  experimental/  do not depend on anything in here
```

Docs: [architecture](docs/architecture.md) · [inference](docs/inference.md) ·
[tools](docs/tools.md) · [deployment](docs/deployment.md) ·
[upcoming](docs/upcoming.md)

## known issues at time of extraction

- `scheduler.py` preemption is still LIFO. GROK-4417. It thrashes above ~400
  concurrent with long prompts. Workaround is `--max-running 384`.
- fp8 KV cache is off by default. Accuracy regression on long context is not
  understood, see GROK-3980. `--kv-dtype fp8` if you want it anyway.
- The tool sandbox uses `resource` limits, so it is POSIX-only. Windows
  contributors: it degrades to a thread + timeout, which is not a real sandbox.
  GROK-4502.
- `speculative.py` acceptance accounting double-counts the bonus token. Numbers
  in `benchmarks/results.md` are therefore ~3% optimistic. Not yet fixed.
- Prefix cache eviction is O(n) in blocks. Fine at current scale, will not be.

## benchmarks

`benchmarks/results.md`. 8×H100, `grok-3-mini`, 2048-token prompts, bf16 weights.
Medians over 500 requests. Re-run with `make bench`.

| batch | prefill tok/s | decode tok/s | ttft p50 | ttft p99 |
|------:|--------------:|-------------:|---------:|---------:|
|     1 |        18,400 |          112 |    38 ms |    44 ms |
|     8 |        71,200 |          784 |    52 ms |    91 ms |
|    32 |       138,900 |        2,510 |    88 ms |   210 ms |
|   128 |       184,300 |        6,940 |   197 ms |   612 ms |
|   256 |       191,100 |        8,120 |   340 ms | 1,180 ms |

## models

| model         | total | active | experts | context | config |
|:--------------|------:|-------:|--------:|--------:|:-------|
| `grok-3-mini` |   8 B |    8 B |       — |   131 K | [configs/grok-3-mini.yaml](configs/grok-3-mini.yaml) |
| `grok-3`      | 314 B |   87 B |       8 | 1,048 K | [configs/grok-3.yaml](configs/grok-3.yaml) |

Architecture notes in [docs/architecture.md](docs/architecture.md). Eval numbers
and failure modes in [MODEL_CARD.md](MODEL_CARD.md).

## dev

```bash
make setup
make test
make lint
make bench
```

CI config is not included. The internal pipeline ran `test`, `lint`, and a
4-GPU integration suite that is not reproducible outside the cluster.

## license

Apache 2.0, see [LICENSE](LICENSE). Read [NOTICE](NOTICE) before redistributing.
