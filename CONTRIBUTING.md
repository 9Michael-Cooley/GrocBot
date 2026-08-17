# Contributing

## Setup

```bash
make setup     # editable install + dev deps
make test
make lint
```

No runtime dependencies. The monorepo forbade third-party deps in
`//research/serving`, which is why there's a vendored YAML subset parser in
`config.py` and a hand-rolled Prometheus exposition in `telemetry/metrics.py`.
Keep it that way — `pytest`, `ruff`, and `mypy` are dev-only.

## Review

`OWNERS` maps paths to teams. `safety/` needs two reviewers, no exceptions.

## Style

Ruff, 100 columns. `make fmt`.

**Comments explain why, not what.** The codebase is full of decisions that look
arbitrary and are not — sampler ordering, RoPE pairing convention, the fp32
router. If you change one of those, move the comment with it. If you find one
that's wrong, fix the comment in the same change.

Don't add a docstring restating the signature. Do add one when the function has
a failure mode a caller needs to know about.

## Tests

Everything in `src/grokbot/` except `experimental/` needs tests.

Write the test against the behavior, not the implementation. A test that asserts
what the code currently does is worth very little; the ones in this repo that
have caught real bugs assert invariants — round-trips reassemble, streaming
matches batched, refcounts never go negative.

Some specific traps this suite has hit:

- **Identical prompts share cache blocks.** A scheduler test that queues the same
  prompt many times creates no cache pressure at all, no matter how many requests
  it adds. `make_request` in `test_scheduler.py` generates distinct prompts by
  default for exactly this reason.
- **The synthetic backend is prompt-determined**, so the sampler seed does not
  change its output. Test sampler behavior against the sampler.
- **Don't assert away a known bug.** `test_thrashing_under_pressure_is_reproducible`
  asserts that GROK-4417 still reproduces. When it's fixed, that test should fail
  and get rewritten, not deleted quietly.

## Known issues

Filed ones are in `TODO` and referenced from the code by ticket. If you fix one,
remove the comment and the TODO entry in the same change — stale "known issue"
comments about fixed bugs cost more time than no comment.

Current: GROK-4417 (LIFO preemption thrashing), GROK-3980 (fp8 KV long-context
regression), GROK-4502 (no sandbox on Windows), and the speculative acceptance
double-count, which is unfiled.

## Performance

Include before/after from `benchmarks/bench_decode.py` for anything touching the
scheduler, cache, or sampler. State the hardware. A change that helps at batch 1
and hurts at batch 256 is not an improvement — sweep it:

```bash
python benchmarks/bench_decode.py --sweep --prompt-tokens 2048
```

Remember that without the CUDA backend this measures Python overhead in the
serving path, not model throughput. That is the right measurement for scheduler
changes and the wrong one for kernel changes.

## What not to do

- Don't add a runtime dependency.
- Don't reorder the sampler pipeline without a side-by-side. Getting it wrong
  produces output that is subtly worse in ways no test catches.
- Don't change the RoPE pairing convention. It is not interchangeable with the
  checkpoints and it fails silently.
- Don't lower `router_dtype` below fp32.
- Don't put anything in `experimental/` that something else imports.
