# Tools

## Defining one

A tool is a function. The schema is generated from its signature.

```python
from grokbot.tools import tool

@tool(description="Look up the current price of a ticker.")
def stock_price(symbol: str, currency: str = "USD") -> float:
    return fetch(symbol, currency)
```

Generated, not hand-written, on purpose: hand-written schemas drift from the
implementation, and the model then calls a signature that doesn't exist.

Everything must be annotated. Unannotated parameters raise at **decoration**
time — at import, not when the model first tries to call it.

### Type mapping

| Python | JSON Schema |
|:--|:--|
| `str` `int` `float` `bool` | `string` `integer` `number` `boolean` |
| `list[T]` | `array` with `items` |
| `dict` | `object` |
| `Literal["a","b"]` | `string` with `enum` |
| `T \| None`, `Optional[T]` | schema for `T`, omitted from `required` |

Anything else raises. Add a mapping in `_schema_for` rather than falling back to
`string` — a silent fallback means the model sends a shape you don't handle.

### Registries

`REGISTRY` is the global default. For isolation, pass your own:

```python
reg = ToolRegistry()

@tool(description="Scoped.", registry=reg)
def scoped(x: int) -> int:
    return x
```

> `ToolRegistry` defines `__len__`, so an **empty registry is falsy**. Inside the
> package, use `registry if registry is not None else REGISTRY` — never
> `registry or REGISTRY`. The `or` form sent every tool decorated with a fresh
> registry into the global one instead. Same trap applies to any code holding a
> possibly-empty registry.

Duplicate names raise unless `replace=True`. Last-one-wins used to be silent,
which shipped a registry where a test double shadowed the real tool in
production for two days.

## Calling

`Tool.validate_args` runs before the function:

- missing required arguments → `ToolError`
- **unknown arguments are dropped**, with a debug log. Models hallucinate
  parameters; failing the whole turn is worse than ignoring one. A persistent
  one usually means the description is misleading.
- scalars are coerced to the declared type
- `enum` membership is enforced

## The sandbox

Every call goes through `tools/sandbox.py`.

**POSIX:** fork, apply `RLIMIT_AS` / `RLIMIT_CPU` / `RLIMIT_NOFILE` / `RLIMIT_CORE`
in the child, run there, pipe the pickled result back. A runaway tool takes down
the child.

**Windows:** there is no fork and no `resource` module. It degrades to a daemon
thread with a join timeout. **This is not a sandbox.** Python cannot kill a
thread, so the thread keeps running past the timeout — a hung tool leaks a thread
per call — and the memory ceiling is not enforced at all. Good enough for local
development, nothing else. (GROK-4502)

Set `strict=True` to refuse rather than pretend:

```python
Sandbox(SandboxPolicy(strict=True, wall_clock_s=10, memory_mb=512))
```

Serving sets this. Honestly the default should be flipped.

## Writing a safe tool

The sandbox bounds resource use. It does not make a tool's *logic* safe.

**Never `eval()`.** `builtin.calculator` walks a parsed AST with an explicit node
allowlist. A calculator is the single most obvious place a prompt-injected
expression lands, and `eval()` there is arbitrary code execution wearing a hat.
Note it also bounds the exponent — `2**(2**30)` is a one-line hang.

**Validate the resolved address, not the hostname.** `http_get` ships disabled
because the egress proxy client isn't in this tree. That proxy does DNS pinning
and blocks link-local and RFC1918 ranges. Without it, a tool call from inside a
serving pod is an SSRF primitive aimed at the metadata endpoint. If you register
your own fetch tool, resolve first and check the address — checking the hostname
lets DNS rebinding straight through.

**Mark destructive tools `dangerous=True`.** They're excluded from
`schemas()` and the agent loop refuses them unless explicitly enabled.

**Treat tool output as untrusted.** It came from the internet, a database, or a
file — all places an attacker can write. `PolicyEngine.check_tool_result` filters
it on the way back in, same as a user turn.

## The agent loop

```python
agent = Agent(engine, tools=registry, persona="default", max_iterations=8)
result = agent.run("what's 17 * 23?")
```

`render memory → generate → parse calls → execute → record → repeat`, until the
model stops emitting calls or a guard trips.

The guards are the point. A loop without them turns a transient tool error into
an unbounded bill — the model retries the same failing call and has no mechanism
to notice.

| Guard | Default | Behavior |
|:--|:--|:--|
| `max_iterations` | 8 | hard ceiling on round trips |
| `max_tool_calls` | 24 | total across all iterations |
| repetition | — | identical `(tool, args)` twice in a row is refused with a message telling the model to vary or use the prior result |
| `max_consecutive_failures` | 3 | ends the loop and reports the last error |

### Wire format

```
<|tool_call|>{"name": "calculator", "arguments": {"expression": "17*23"}}<|/tool_call|>
```

`parse_tool_calls` skips malformed JSON rather than raising — one bad call
shouldn't lose a turn that also contains good ones — and salvages the
unterminated case, which is what you get when a call is cut off by `max_tokens`.

The protocol text in the system prompt is generated from live schemas
(`persona.tool_protocol`). Hand-writing it means the prompt describes tools that
don't exist, which the model then tries to call. That was the top source of
malformed calls before it was generated.
