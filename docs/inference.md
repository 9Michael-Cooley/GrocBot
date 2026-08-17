# Inference

Practical notes on getting output you want.

## Sampling

```python
from grokbot import Engine, GenerationConfig

engine = Engine.from_config("configs/grok-3-mini.yaml")
gen = GenerationConfig(temperature=0.7, top_p=0.95, top_k=64, max_tokens=1024)

completion = engine.generate("explain paged attention", gen)
print(completion.text, completion.usage())
```

### Choosing parameters

| Task | temperature | top_p | notes |
|:--|--:|--:|:--|
| Extraction, classification | 0.0 | — | greedy; reproducible |
| Code | 0.2–0.4 | 0.95 | higher invents APIs |
| General chat | 0.7 | 0.95 | default |
| Creative | 0.9–1.1 | 0.98 | add `repetition_penalty` 1.1 |

`temperature=0` is exactly greedy and ignores `top_p`/`top_k` entirely — the
sampler short-circuits.

Below ~0.3 with no repetition penalty, open-ended prompts loop. That's the model,
not the sampler.

### min_p

Often better than top-p. It sets the cutoff at `min_p × p_max`, so it scales with
how confident the step is: prunes hard when the model is sure, barely at all when
it isn't. Try `min_p=0.05` with `top_p=1.0`.

### Penalties

Three, and they do different things:

- `repetition_penalty` (multiplicative, ~1.05–1.15) — divides positive logits,
  multiplies negative ones. Blunt but effective.
- `frequency_penalty` (additive, scales with count) — targets overused tokens.
- `presence_penalty` (additive, flat once seen) — pushes toward new topics.

All are bounded by `penalty_window` (default 256). Unbounded, long conversations
accumulate penalty on common words until the model can no longer write "the".

## Determinism

Same seed + same prompt + same config → same output.

```python
GenerationConfig(temperature=0.9, seed=1234)
```

The RNG is our own splittable generator (`utils/rng.py`), not `random`, precisely
so nothing else in the process can perturb the stream.

Greedy (`temperature=0`) is deterministic without a seed.

> Under `SyntheticBackend` the seed has no visible effect: the stand-in derives
> its stream from a hash of the prompt and emits a near-argmax spike. That's the
> fake backend, not the sampler.

## Streaming

```python
for chunk in engine.stream(prompt, gen):
    print(chunk.text, end="", flush=True)
    if chunk.is_final:
        print(f"\n[{chunk.finish_reason}]")
```

Exactly one chunk has `is_final`. Some chunks have empty `text` — a token that
completes only part of a character, or text held pending a possible stop string.
Don't treat empty text as end-of-stream.

Streaming and batched output are identical for the same seed. There's a test.

## Stop conditions

`FinishReason` tells you which fired:

| Value | Meaning |
|:--|:--|
| `LENGTH` | hit `max_tokens` |
| `STOP` | a `stop` string matched (not included in output) |
| `STOP_TOKEN` | EOS / `<\|im_end\|>` (not rendered) |
| `TOOL_CALL` | stopped to call a tool |
| `CANCELLED` | aborted |

Stop strings are truncated *before* the match. At most 8, and empty strings are
ignored — an empty stop matches everywhere and would pin the buffer forever.

## Prompt caching

On by default. Requests sharing a prefix share cache blocks, so a long system
prompt is paid for once.

To benefit: **put the stable part first.** System prompt, then few-shot
examples, then the variable part. A timestamp at the top defeats it entirely.

Only *full* blocks (16 tokens) are shareable — a partial block breaks the chain.
`completion.cached_tokens` reports what was reused.

## Context

```python
engine.model_config.max_context()   # min(max_position_embeddings, rope reach)
```

`prompt + max_tokens` must fit, checked up front — `ContextLengthExceeded` before
any compute is spent.

Retrieval degrades in the last ~15% of the window on both models. Near the limit,
put what matters at the start or the end, not the middle.

## Speculative decoding

```yaml
speculative:
  draft_model: grok-3-draft-1b
  num_tokens: 5
  acceptance: typical
```

`typical` preserves the target distribution exactly. `greedy` matches argmax and
is bit-identical to non-speculative greedy, but only valid at temperature 0.

More draft tokens is not better — draft cost eventually dominates:

```python
from grokbot.inference.speculative import optimal_depth, theoretical_speedup
optimal_depth(acceptance=0.7)              # -> 4
theoretical_speedup(0.7, k=5)              # -> 1.96
```

Measured, n=5 beats n=8. Acceptance depends on how well the draft matches the
target; a draft from the same family does much better than a generic small model.

> Reported acceptance includes the bonus token and reads ~`1/(k+1)` high.
> `benchmarks/results.md` inherits the error.

## Throughput

```bash
python -m grokbot bench --requests 128 --prompt-tokens 2048 --output-tokens 128
```

Levers, in the order worth trying:

1. **Batch size.** Decode is memory-bandwidth bound; larger batches are close to
   free until the cache runs out. Raise `max_running` until preemptions appear,
   then back off.
2. **Prefix caching.** Free if prompts share structure.
3. **fp8 KV.** Doubles concurrency, see GROK-3980.
4. **Speculative decoding.** ~2× on decode, costs a second model's memory.
5. **Chunked prefill.** Trades a little prefill throughput for much better p99.

Watch `grokbot_preemptions_total`. A nonzero sustained rate means you are past
the useful batch size and are now paying to recompute work you already did.
