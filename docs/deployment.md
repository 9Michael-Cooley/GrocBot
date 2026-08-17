# Deployment

## Before anything else

This tree does not include the CUDA backend, so it will not serve a real model.
`load_backend` falls back to `SyntheticBackend` and logs a warning. If you are
reading this to deploy something, you need `model/backend_cuda.py` first.

What follows is the operational shape that was in use, kept because the config
files reference it and because the sizing arithmetic is still correct.

## Sizing

The two questions are: do the weights fit, and how much is left for KV.

```bash
python -m grokbot info --config configs/grok-3.yaml
```

```
params total       313.8B
params active      87.3B
weights (bf16)     584.5 GiB
kv per token       256.0 KiB
```

584.5 GiB of weights means `grok-3` needs TP ≥ 8 on 80 GiB cards, and that is
before any KV. `TransformerConfig.plan_kv_blocks` does the division:

```
budget    = device_memory × gpu_memory_utilization
free      = budget − weights/tp − 8% activation reserve
num_blocks = free / (kv_bytes_per_token × block_size / tp)
```

The activation reserve is a flat fraction, not an estimate. The real number
depends on batch composition you haven't seen yet, and overshooting OOMs
mid-serve. Leave it conservative.

`tensor_parallel` cannot exceed `num_kv_heads` (8 on both models) — `validate()`
rejects it. KV heads are what gets sharded.

## Configuration layering

Configs inherit with `extends`:

```yaml
extends: grok-3-mini.yaml
server:
  port: 8080
```

Child deep-merges over parent. Then `GROKBOT_*` environment variables override
everything:

```bash
export GROKBOT_WEIGHTS=/mnt/checkpoints/grok-3
export GROKBOT_SCHEDULER__MAX_RUNNING=384      # __ is the path separator
export GROKBOT_TELEMETRY__LOG_LEVEL=debug
```

Then CLI flags. Do not edit the reference configs in place; copy to
`configs/local.*`, which is gitignored.

## Settings that matter

| Setting | Default | Notes |
|:--|:--|:--|
| `scheduler.max_running` | 256 | **Keep ≤ 384 on `grok-3`.** Higher thrashes (GROK-4417). |
| `scheduler.chunk_size` | 2048 | Lower = better p99, worse prefill throughput. |
| `scheduler.max_batched_tokens` | 8192 | Per-step budget across prefill + decode. |
| `cache.block_size` | 16 | Power of two. Larger wastes more per sequence. |
| `cache.prefix_caching` | true | Large win on shared system prompts. |
| `cache.dtype` | auto | `fp8` halves KV but see GROK-3980. |
| `runtime.gpu_memory_utilization` | 0.90 | Above 0.95 OOMs on long-tail batches. |

### fp8 KV

`--kv-dtype fp8` halves cache memory and roughly doubles concurrency. There is
an unexplained accuracy regression on long context (GROK-3980). Arch believes
it's the router logits, not the cache — reproduce on `grok-3-mini`, which has no
router, to rule that out. Off by default.

## Running

```bash
python -m grokbot serve --config configs/serving.yaml --host 0.0.0.0 --port 8080
```

Endpoints:

| Path | Purpose |
|:--|:--|
| `POST /v1/chat/completions` | OpenAI-compatible, `stream: true` for SSE |
| `POST /v1/completions` | legacy prompt completion |
| `GET /v1/models` | served model names |
| `GET /health` | liveness |
| `GET /metrics` | Prometheus |
| `GET /stats` | scheduler, cache, limiter, policy counters |

### The stdlib server

`serve/api.py` holds the engine lock for the duration of each request, so
**effective concurrency is 1** and the scheduler's continuous batching is
invisible from outside. That is a property of this server, not the engine — the
real ASGI adapter drives the same step loop from one thread and multiplexes
streams off it. It is in `//platform/api` and is not here.

Also missing: SSE backpressure. A slow reader blocks the write and pins the
engine lock for everyone.

Do not put this in front of real traffic.

## Auth and limits

`api.api_keys: []` means **no authentication**. In the cluster the edge
terminated auth and this was intentional; standalone it is an open endpoint.

`limits.*` is a per-process token bucket. With `server.workers > 1` or more than
one replica, the effective limit is N× what is configured. It is defence in
depth behind the edge limiter, not the control. Fixing it needs shared state.

## Safety

```yaml
safety:
  enabled: true
  input_filters: [pii, injection]
  output_filters: [pii, leakage]
  on_block: refuse
  on_classifier_error: block     # NOT the default — set this
```

`PolicyEngine.classify` calls a service that isn't in this tree and **fails open**
by default. Fail-open is right for a local dev tree and wrong for everything
else. Set `on_classifier_error: block`.

`telemetry.log_prompts` must stay false outside staging.

## Observability

Key metrics:

| Metric | Watch for |
|:--|:--|
| `grokbot_ttft_seconds` | p99 climbing → prefill contention, lower `chunk_size` |
| `grokbot_preemptions_total` | any sustained rate → past the thrash point |
| `grokbot_kv_cache_utilization` | pinned fraction; sustained >0.95 → cache-bound |
| `grokbot_waiting_sequences` | growing → shed load upstream |
| `grokbot_requests_failed_total` | 503s are cache exhaustion, not bugs |

Histogram buckets are dense in the 20–500 ms band because the default client
buckets put four boundaries above 1 s and two below 100 ms — exactly backwards
for TTFT, and the p99 ends up interpolated out of nothing.

Tracing defaults off and its collector endpoint is stripped.

## Container

```bash
docker build -t grokbot:0.4.2 .
docker run --rm -p 8080:8080 -v /mnt/checkpoints:/weights:ro \
  -e GROKBOT_WEIGHTS=/weights/grok-3-mini grokbot:0.4.2
```

The image has no CUDA runtime — it will run synthetic. A real image needs the
kernel module and a matching CUDA base.

## Diagnosis

| Symptom | Cause |
|:--|:--|
| "no weights configured", garbage output | expected; no checkpoint |
| `CudaBackend is unavailable` | kernels not in this tree |
| Rising `preemptions_total` | lower `max_running` (GROK-4417) |
| 503 `OutOfCacheBlocks` | cache too small for the concurrency; lower `max_running` or raise `num_blocks` |
| p99 TTFT spikes, p50 fine | long prefills; lower `chunk_size` |
| Tokenizer errors on load | `tokenizer.json` missing from the checkpoint dir |
| Rate limits ~N× too loose | per-process limiter, N replicas |
| Tools hang forever | Windows; there is no real timeout (GROK-4502) |
