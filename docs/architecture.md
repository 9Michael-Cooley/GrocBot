# Architecture

How a request becomes tokens, and why each piece is shaped the way it is.

## Request path

```
HTTP /v1/chat/completions
  └─ serve/protocol.py     parse, coerce OpenAI shape -> GenerationConfig
  └─ serve/ratelimit.py    token bucket + concurrency guard
  └─ safety/policy.py      input filters
  └─ inference/engine.py   ──► scheduler.add_request()
                              │
                              ├─ scheduler.schedule()   build this step's batch
                              ├─ backend.forward_batch()   logits
                              ├─ sampler.sample()          token
                              └─ stream.StreamAssembler    text
  └─ safety/policy.py      output filters
  └─ serve/api.py          SSE frames
```

The step loop in `Engine._step` is the only place sequence state changes. That
constraint is what makes the scheduler debuggable — everything else reads.

## Attention: GQA

Query heads are grouped, and each group shares one KV head. At 32 query heads
and 8 KV heads the cache is 4× smaller than full multi-head attention.

The tradeoff is quality against cache size, and the cache is what bounds
concurrency. At 128 K context on `grok-3-mini`:

| Scheme | KV heads | Bytes/token | 128 K context |
|:--|--:|--:|--:|
| MHA | 32 | 512 KiB | 64 GiB |
| GQA 4:1 | 8 | 128 KiB | 16 GiB |
| MQA | 1 | 16 KiB | 2 GiB |

MHA at 128 K is most of an H100 for one sequence. MQA measurably hurts. 4:1 was
where the quality difference stopped being detectable on our evals.

`AttentionConfig.group_size` is the map: query head `h` reads KV head
`h // group_size`.

## Position: RoPE + YaRN

Rotary embeddings encode position as a rotation applied to Q and K, so attention
scores depend on relative distance rather than absolute index.

Training at 1 M context is not affordable, so the models train at 32 K and the
frequencies are stretched at inference. Two options:

- **Linear (PI)** divides every frequency by the scale factor. Simple, and it
  degrades local structure — adjacent tokens become harder to distinguish.
- **YaRN** interpolates only the low-frequency dimensions and leaves
  high-frequency ones alone. Local structure survives. It also applies an
  attention temperature correction (`_yarn_mscale`), without which long context
  degrades even when positions interpolate correctly.

`grok-3` uses YaRN at ×32 over a 32 K base.

**The pairing convention matters.** `rope.rotate` pairs dimension `i` with
`i + d/2` (split-half). Pairing adjacent elements is also valid RoPE, but is not
interchangeable with these weights — it degrades quality silently rather than
failing. That cost a week once.

## FFN: MoE on `grok-3`

Each token is routed to 2 of 8 experts, so 313.8 B parameters exist but 87.3 B
run per token.

`model/moe.py` implements token-choice top-k routing with capacity limits:

- **Capacity** = `ceil(tokens × k / experts × capacity_factor)`. Without it, one
  hot expert serializes the whole batch.
- **Overflow spills** to the next-best expert rather than dropping the token.
  Dropping a token entirely is much worse than routing it slightly wrong.
- **Router runs in fp32.** Lowering it changes which expert wins on near-ties,
  and the resulting quality drift is subtle enough that it took weeks to
  attribute. `config.validate()` rejects anything else. See GROK-3980.
- **Load imbalance** (`max/mean`) is the health metric. Sustained above ~1.5
  means a routing problem.

`aux_loss` is dead at inference — kept for checkpoint schema compatibility and
because the eval harness reports it when replaying training batches.

## Memory: paged KV cache

`model/kv_cache.py`. Attention state lives in fixed-size blocks (16 tokens) and
each sequence holds a block table.

Contiguous allocation requires reserving `max_tokens` up front for every
sequence. Most sequences finish far short of it, so most of that reservation is
never touched — the effective utilization we measured before paging was under
30%. Paging bounds waste at `block_size - 1` tokens per sequence.

Blocks are **refcounted**, which gives three things for free:

- **Prefix sharing.** Full blocks are hashed over their contents plus the
  preceding block's hash. Two requests with the same prefix share the blocks.
  A partial block breaks the chain — it cannot be shared because its future
  contents are undetermined.
- **Copy-on-write forking.** `fork()` shares the parent's blocks; the first
  write to a shared block copies it.
- **Deferred eviction.** Freeing a sequence returns partial blocks to the pool
  but leaves full blocks populated and unreferenced, so a later request can
  still hit them. They are reclaimed on demand, LRU.

That last point is why `can_allocate` counts free **and** evictable blocks.
Counting only the free list made allocation fail while reclaimable capacity sat
idle, and the scheduler answered by preempting a live sequence for nothing.

Eviction is currently an O(n) scan over all blocks. Fine at this scale, won't be.

## Scheduling

`inference/scheduler.py`. Continuous batching: a request joins the running batch
the moment a slot frees, rather than waiting for the batch to drain.

Each step spends two budgets — tokens (`max_batched_tokens`) and cache blocks —
in this order:

1. **Decodes.** Already hold their blocks, need one slot each. Cheap, and
   starving them is what users feel as stutter.
2. **In-flight prefills.** Already holding blocks; finishing frees the slot
   sooner than starting something new.
3. **New admissions**, FIFO from the waiting queue.

**Chunked prefill** splits long prompts so a 100 K-token prefill cannot stall
every decoding sequence behind it. That head-of-line block was the single worst
p99 contributor before chunking landed.

**Preemption is LIFO and this is a known bug (GROK-4417).** The newest sequence
is evicted. Above ~400 concurrent with long prompts it thrashes: the victim is
re-admitted, preempts someone else, and recompute cost cascades. Workaround is
`--max-running 384`. The fix is a priority queue keyed on
`(arrival, tokens_done)` so nearly-finished sequences stop being evicted.

Preemption is **recompute**: KV is discarded, but generated tokens are folded
into the prompt so the retry regenerates the same state. Swap-to-host was never
finished and the constructor rejects it rather than half-working.

## Sampling

`inference/sampler.py`. Order is not arbitrary:

```
logit bias -> suppression -> penalties -> temperature -> top-k -> top-p -> min-p -> draw
```

Penalties precede temperature so raising temperature doesn't quietly rescale how
hard the penalty bites. Truncation follows temperature so top-p measures the
distribution the caller actually asked for. Reordering produces output that is
subtly worse in ways no test catches.

Two details worth knowing:

- Repetition penalty **divides** positive logits and **multiplies** negative
  ones. Multiplying throughout would reward tokens whose logit is negative.
- `min_p` scales its cutoff with the peak probability, so it prunes hard on
  confident steps and barely at all on uncertain ones. Often better than top-p.

## Streaming

Two things straddle token boundaries and both are handled in `inference/stream.py`.

**Stop strings.** If `END` arrives as `E` + `ND`, emitting the `E` is
unrecoverable — it's on the wire. `StopChecker` holds any suffix that could still
become a stop string and releases it once resolved.

**Multi-byte characters.** `Tokenizer.decode_incremental` decodes a suffix window
and diffs against the previous rendering, stripping trailing U+FFFD from *both*
sides. Stripping only one side desynchronizes the prefix length and eats real
characters on the token that completes the sequence.

## Speculative decoding

`inference/speculative.py`. A draft model proposes k tokens; the target verifies
all k in one pass. Decode is memory-bandwidth bound, so verifying k tokens costs
about what decoding one costs.

`typical` acceptance (accept with probability `min(1, p_target/p_draft)`, sample
rejections from the normalized residual) preserves the target distribution
exactly. `greedy` matching does not, once temperature > 0.

Depth has an optimum — past it, draft cost dominates. `optimal_depth()` computes
it; measured, n=5 beats n=8.

**Known: `num_accepted` counts the bonus token**, so acceptance reads high by
roughly `1/(k+1)` and `benchmarks/results.md` is ~3% optimistic.

## What isn't here

- `model/backend_cuda.py` — the kernels. Never opened to the research org.
- The ASGI adapter (`//platform/api`). The stdlib server here holds the engine
  lock per request, so concurrency is effectively 1 and continuous batching is
  invisible from outside. That's this server, not the engine.
- The policy classifier client.
- The egress proxy client that makes `http_get` safe.
- `tokenizer.json` — ships with the checkpoint.
