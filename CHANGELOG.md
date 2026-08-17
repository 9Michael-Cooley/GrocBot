# Changelog

Internal versioning. Dates are cluster rollout, not tag dates.

## Unreleased — 0.5.0 candidate

Nothing here is shipped. See [docs/upcoming.md](docs/upcoming.md).

- **Pets** (GROK-4590). Companion bots: pick a dog (defaults to **Odie**) or a
  cat (defaults to **Garfield**). Hunger, energy, and affection decay on a wall
  clock and are folded into the system prompt each turn, so state changes the
  answer. Interactions are feed / play / pet / nap; the cat weights food far
  higher than play and its affection decays 4.5× faster than the dog's.
  Behind `GROKBOT_ENABLE_PETS=1`.
  Blocked on persistence (GROK-4611) — state currently dies with the process.
  Households are stubbed, pets can't perceive each other. Safety has not
  reviewed the personas. Bird was cut from scope; the profile is still in the
  file and gated out.

## 0.4.2

Added
- **Bot presets** (`agent/presets.py`): `default`, `fast`, `deep`, `code`,
  `research`, `creative`, `fun`, and hidden `eval`. Bundles persona, sampling,
  tool access, and memory policy, which were previously assembled per-caller and
  drifted — the `code` surface in one product ran at temperature 0.7 for six
  weeks because it inherited the chat default.
- `grokbot presets`, and `--preset` on `chat` and `agent`.

Fixed
- Tool schema derivation raised on `X | None` annotations. `types.UnionType` and
  `typing.Union` are different objects and `get_origin` reports each as itself;
  only the latter was handled.
- `@tool(registry=...)` registered into the global registry instead. `ToolRegistry`
  defines `__len__`, so an empty registry is falsy and `registry or REGISTRY`
  took the wrong branch every time.
- `can_allocate` counted only the free list, ignoring unreferenced cached blocks
  that are reclaimable on demand. Allocation failed with capacity available, and
  the scheduler responded by preempting a live sequence for nothing.
- `PagedKVCache.utilization` counted cached-but-unreferenced blocks as in use, so
  the gauge pinned at 100% once prefix caching warmed up.
- Working memory summary grew without bound, eventually driving the token budget
  negative and evicting the entire history on the next turn. Now capped at
  `max_tokens // 8`, trimmed by tokens rather than characters.
- PII card pattern consumed the separator following the number, running redacted
  output into the next word.

## 0.4.1

Fixed
- Streaming dropped characters mid-emoji. `decode_incremental` stripped trailing
  U+FFFD from only one side of the diff, desynchronizing the prefix length on the
  token that completed the sequence.
- Pretokenizer silently dropped underscores and unpaired surrogates — the split
  pattern was not total, and unmatched characters vanished from `findall`.

## 0.4.0

Added
- Chunked prefill (`scheduler.chunked_prefill`). 13× p99 improvement on mixed
  workloads for a 9% prefill throughput cost. Head-of-line blocking behind long
  prefills was the worst tail contributor.
- `min_p` sampling.
- Prometheus `/metrics` and `/stats`.
- Agent loop guards: repetition detection, consecutive-failure limit, tool-call
  budget.

Changed
- Histogram buckets rewritten. The defaults put four boundaries above 1 s and two
  below 100 ms, which is backwards for TTFT — everything landed in one bucket and
  p99 was interpolated out of nothing.
- Duplicate tool registration now raises instead of last-one-wins. A test double
  shadowed a real tool in production for two days.

Known
- GROK-4417: LIFO preemption thrashes above ~400 concurrent. `max_running: 384`.

## 0.3.0

Added
- MoE routing with capacity limits and overflow spill.
- Speculative decoding, greedy and typical acceptance.
- YaRN rope scaling; `grok-3` to 1 M context.

Changed
- Router forced to fp32. Lowering it flips near-tie expert selection and the
  quality drift took weeks to attribute (GROK-3980).
- Capacity overflow spills to the next-best expert rather than dropping the
  token. Dropping is much worse.

Known
- Speculative acceptance rate counts the bonus token and reads ~`1/(k+1)` high.
  `benchmarks/results.md` inherits the error.

## 0.2.0

Added
- Paged KV cache with refcounting, prefix sharing, and copy-on-write forking.
  Measured utilization before paging was under 30%.
- Continuous batching.
- OpenAI-compatible API.

Removed
- Contiguous cache allocation.

## 0.1.0

Initial extraction from `//research/serving`. Dense models only, static batching,
no tool support.
