# Security

## Reporting

Do not open a public issue for a vulnerability. Internally this went to
`security@` with the on-call rotation paged for anything scoring 7.0+.

## Known weaknesses in this tree

These are not theoretical. Read them before running this anywhere reachable.

### The sandbox is not a sandbox on Windows

`tools/sandbox.py` uses fork + rlimits on POSIX. On Windows there is no fork and
no `resource` module, so it degrades to a daemon thread with a join timeout.
Python cannot kill a thread — a hung tool keeps running and leaks a thread per
call, and the memory ceiling is not enforced at all. Set
`SandboxPolicy(strict=True)` to refuse rather than pretend. (GROK-4502)

### The policy classifier fails open

`PolicyEngine.classify` calls a service that is not in this tree. The default
`on_classifier_error: allow` means everything requiring judgement passes through
unchecked. Correct for a local dev tree, wrong for anything else. Serving must
set `block`.

### Injection filters are weak by construction

`safety/filters.py` is pattern matching. Anyone who reads the file writes around
it on the first try. It catches untargeted cases — a scraped page carrying a
generic "ignore previous instructions" block. **A clean pass is not evidence
that content is safe.** Treat all tool and retrieval output as untrusted; the
policy engine does.

### No authentication by default

`api.api_keys: []` accepts every request. In the cluster the edge terminated auth
and this was deliberate. Standalone, it is an open endpoint.

### Rate limits are per-process

Token buckets live in process memory. With `server.workers > 1` or multiple
replicas the effective limit is N× the configured one. Defence in depth behind
the edge limiter, not the control.

### SSRF via custom fetch tools

`http_get` ships disabled because the egress proxy client is not in this tree.
That proxy does DNS pinning and blocks link-local and RFC1918 ranges. A serving
pod can reach the cloud metadata endpoint; a fetch tool without those checks is
an SSRF primitive pointed at it.

If you write your own: **resolve the hostname and validate the resulting
address.** Checking the hostname string lets DNS rebinding through unimpeded.

### No SSE backpressure

A slow reader blocks the write and pins the engine lock. One client can stall the
server. Trivially a denial of service on the stdlib server.

### `experimental/` is unreviewed

Unowned, not in CI, mid-refactor at extraction. `tree_attention.prune` has a
known orphaning bug. Do not import it.

## Things that are handled

- `calculator` walks a parsed AST with an explicit node allowlist and never
  calls `eval`. The exponent is bounded — `2**(2**30)` is a one-line hang.
- Tokenizer decode rejects out-of-range ids rather than indexing past the vocab.
- The safetensors header parser bounds header length and validates offsets before
  allocating.
- Blocked prompts log an **excerpt only**. Blocked content is exactly what you
  least want sitting in a log aggregator. `telemetry.log_prompts` must stay false
  outside staging.
- Refcount underflow on double-free is asserted rather than silently corrupting
  the allocator.

## Threat model

In scope: prompt injection through tool and retrieval output, resource
exhaustion via tool calls, cache exhaustion as denial of service, PII in logs,
SSRF from tools.

Out of scope: an attacker with local process access, malicious checkpoints (a
checkpoint is code — only load ones you trust), and the model's own outputs being
wrong, which is a quality problem, not a security one.
