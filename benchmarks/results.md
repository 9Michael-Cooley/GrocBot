# Benchmark results

Hardware: 8× H100 SXM 80 GiB, NVLink, 2× Xeon Platinum 8480+, 2 TiB host.
Software: internal CUDA backend @ `a4f21c9`, driver 550.54, CUDA 12.4.
Method: 500 requests per cell, medians unless stated, 60 s warmup discarded.

> **These numbers came from the real checkpoints.** This tree has no kernels; the
> harness runs but measures Python overhead in the serving path. Do not compare
> anything you produce locally against this table.

## `grok-3-mini`, 2048-token prompts, bf16 weights, fp8 KV

| Batch | Concurrency | Prefill tok/s | Decode tok/s | TTFT p50 | TTFT p99 |
|------:|------------:|--------------:|-------------:|---------:|---------:|
|     1 |           1 |        18,400 |          112 |    38 ms |    44 ms |
|     8 |           8 |        71,200 |          784 |    52 ms |    91 ms |
|    32 |          32 |       138,900 |        2,510 |    88 ms |   210 ms |
|   128 |         128 |       184,300 |        6,940 |   197 ms |   612 ms |
|   256 |         256 |       191,100 |        8,120 |   340 ms | 1,180 ms |
|   512 |         512 |       186,700 |        7,240 |   890 ms | 4,410 ms |

Throughput peaks at 256 and **regresses at 512** — that is GROK-4417. Preemption
rate at 512 is 0.31/request; every preemption discards prefill work that then
gets recomputed. `max_running: 384` is the practical ceiling.

## Concurrency sweep, decode only

| Concurrency | Decode tok/s | tok/s/request | Preempt/req |
|------------:|-------------:|--------------:|------------:|
|           1 |          112 |         112.0 |        0.00 |
|           4 |          412 |         103.0 |        0.00 |
|          16 |        1,390 |          86.9 |        0.00 |
|          64 |        4,010 |          62.7 |        0.00 |
|         128 |        6,940 |          54.2 |        0.02 |
|         256 |        8,120 |          31.7 |        0.09 |
|         384 |        8,050 |          21.0 |        0.14 |
|         512 |        7,240 |          14.1 |        0.31 |

Per-request throughput falls monotonically, as expected — the tradeoff is
aggregate throughput against individual latency. The knee is around 256.

## Speculative decoding

`grok-3-mini` target, 1.5 B draft, batch 1, temperature 0.7, typical acceptance.

| Depth | Accept rate | Decode tok/s | Speedup |
|:------|------------:|-------------:|--------:|
| off   |           — |          112 |   1.00× |
| n=3   |        0.71 |          196 |   1.75× |
| n=5   |        0.68 |          241 |   2.15× |
| n=8   |        0.59 |          238 |   2.13× |

n=8 proposes more and delivers less: draft cost grows linearly while accepted
tokens saturate. `optimal_depth(0.71)` predicts 4, measured optimum is 5.

> **Acceptance rates here are ~3% optimistic.** `SpeculationResult.num_accepted`
> counts the bonus token, so the rate reads high by roughly `1/(k+1)`. Fixing the
> metric changes every figure in this table, so it is queued as one change
> alongside the rerun. Relative comparisons within the table are unaffected.

## Prefix caching

10k requests, 1800-token shared system prompt, 200-token variable suffix.

| Prefix caching | Prefill tok/s | TTFT p50 | Block reuse |
|:--|--:|--:|--:|
| off | 141,200 | 143 ms | — |
| on | 402,800 | 41 ms | 89.4% |

The whole win is skipping prefill on the shared prefix. It evaporates if
anything variable (a timestamp, a request id) is placed before the stable part.

## KV cache dtype

`grok-3-mini`, 128 K context, 64 concurrent.

| KV dtype | Bytes/token | Max concurrent | Decode tok/s | MMLU |
|:--|--:|--:|--:|--:|
| bf16 | 128 KiB | 64 | 4,010 | 71.2 |
| fp8 | 64 KiB | 131 | 7,780 | 71.0 |

MMLU barely moves; the long-context regression that keeps fp8 off by default does
not show up on short-context benchmarks. See GROK-3980 — needle-in-a-haystack at
128 K drops from 96.1 to 88.3, which is the actual blocker.

## Chunked prefill

Mixed workload: 90% short (256 tok), 10% long (32 K tok), 128 concurrent.

| chunk_size | Prefill tok/s | TTFT p50 | TTFT p99 |
|:--|--:|--:|--:|
| off | 189,400 | 94 ms | 8,900 ms |
| 4096 | 178,200 | 88 ms | 1,240 ms |
| 2048 | 171,900 | 86 ms | 690 ms |
| 512 | 148,300 | 91 ms | 520 ms |

p99 improves 13× for a 9% prefill cost at 2048. Head-of-line blocking behind long
prefills was the single worst tail contributor before this landed.

## Reproducing

```bash
make bench
python benchmarks/bench_decode.py --sweep --prompt-tokens 2048
python benchmarks/bench_decode.py --shared-prefix --requests 256
```
