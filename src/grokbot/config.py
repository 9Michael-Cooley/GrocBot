"""Configuration loading.

Includes a vendored YAML subset parser. The monorepo build did not allow
third-party runtime deps in //research/serving, so rather than take pyyaml we
parse the subset we actually use: nested maps, block and inline lists, scalars,
comments, and anchleless plain documents. No anchors, no multi-doc, no folded
scalars, no tags. If a config needs any of that, the config is wrong.

Configs may declare `extends: <file>` to inherit; the child is deep-merged over
the parent. One level of inheritance is what we use in practice, but it recurses.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .errors import ConfigError

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "none", "~", ""}

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


# --------------------------------------------------------------------------
# scalar / line handling
# --------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Remove a trailing # comment, respecting quotes."""
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote and (i == 0 or line[i - 1] != "\\"):
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _scalar(raw: str) -> Any:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    low = s.lower()
    if low in _NULL:
        return None
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return s


def _inline_seq(raw: str) -> list[Any]:
    """Parse [a, b, "c, d"] — one level, no nesting. We never nest inline."""
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    items, buf, quote = [], [], None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    return [_scalar(x) for x in items if x.strip() != ""]


def _value(raw: str) -> Any:
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        return _inline_seq(s)
    if s.startswith("{"):
        raise ConfigError("inline maps are not supported by the vendored parser")
    return _scalar(s)


# --------------------------------------------------------------------------
# block parser
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ConfigError(f"tab indentation at line {lineno}; use spaces")
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        rows.append((len(stripped) - len(stripped.lstrip()), stripped.strip()))
    return rows


def _parse_block(rows: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    if i >= len(rows):
        return None, i

    if rows[i][1].startswith("- "):
        seq: list[Any] = []
        while i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- "):
            body = rows[i][1][2:].strip()
            if ":" in body and not body.startswith("["):
                # list of maps: re-parse the item as a nested block
                sub = [(indent + 2, body)]
                j = i + 1
                while j < len(rows) and rows[j][0] > indent:
                    sub.append(rows[j])
                    j += 1
                val, _ = _parse_block(sub, 0, indent + 2)
                seq.append(val)
                i = j
            else:
                seq.append(_value(body))
                i += 1
        return seq, i

    mapping: dict[str, Any] = {}
    while i < len(rows) and rows[i][0] == indent:
        line = rows[i][1]
        if ":" not in line:
            raise ConfigError(f"expected 'key: value', got {line!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest:
            mapping[key] = _value(rest)
            i += 1
        else:
            child_indent = rows[i + 1][0] if i + 1 < len(rows) else indent
            if child_indent <= indent:
                mapping[key] = None  # `key:` with nothing under it
                i += 1
            else:
                mapping[key], i = _parse_block(rows, i + 1, child_indent)
    return mapping, i


def parse_yaml(text: str) -> dict[str, Any]:
    rows = _tokenize(text)
    if not rows:
        return {}
    value, _ = _parse_block(rows, 0, rows[0][0])
    if not isinstance(value, dict):
        raise ConfigError("top level of a config must be a mapping")
    return value


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dotted-path view over a parsed config tree."""

    def __init__(self, data: dict[str, Any], source: str | None = None):
        self._data = data
        self.source = source

    # -- access ------------------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        val = self.get(path, sentinel)
        if val is sentinel:
            raise ConfigError(f"missing required key {path!r} in {self.source or '<config>'}")
        return val

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {path!r}: {part!r} is a scalar")
        node[parts[-1]] = value

    def section(self, path: str) -> Config:
        val = self.get(path, {}) or {}
        if not isinstance(val, dict):
            raise ConfigError(f"{path!r} is not a section")
        return Config(val, source=self.source)

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def __contains__(self, path: str) -> bool:
        sentinel = object()
        return self.get(path, sentinel) is not sentinel

    def __getitem__(self, path: str) -> Any:
        return self.require(path)

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, keys={sorted(self._data)})"

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, _seen: set[str] | None = None) -> Config:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config not found: {p}")

        seen = _seen or set()
        key = str(p.resolve())
        if key in seen:
            raise ConfigError(f"circular extends chain at {p}")
        seen.add(key)

        data = parse_yaml(p.read_text(encoding="utf-8"))

        parent = data.pop("extends", None)
        if parent:
            parent_path = (p.parent / str(parent)).resolve()
            base = cls.load(parent_path, _seen=seen)
            data = deep_merge(base.as_dict(), data)

        cfg = cls(data, source=str(p))
        cfg._apply_env()
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(dict(data), source="<dict>")

    def _apply_env(self) -> None:
        """GROKBOT_* env overrides. GROKBOT_SCHEDULER__MAX_RUNNING=64 etc."""
        for name, raw in os.environ.items():
            if not name.startswith("GROKBOT_"):
                continue
            if name == "GROKBOT_WEIGHTS":
                self.set("weights.path", raw)
                continue
            path = name[len("GROKBOT_") :].lower().replace("__", ".")
            self.set(path, _scalar(raw))

    def apply_overrides(self, overrides: dict[str, Any]) -> Config:
        for k, v in overrides.items():
            if v is not None:
                self.set(k, v)
        return self

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        heads = self.get("model.num_heads")
        kv_heads = self.get("model.num_kv_heads")
        if heads and kv_heads and heads % kv_heads:
            raise ConfigError(f"num_heads ({heads}) must be divisible by num_kv_heads ({kv_heads})")

        block = self.get("cache.block_size", 16)
        if block & (block - 1):
            raise ConfigError(f"cache.block_size must be a power of two, got {block}")

        moe = self.get("model.moe")
        if moe:
            n, k = moe.get("num_experts", 0), moe.get("experts_per_token", 0)
            if k > n:
                raise ConfigError(f"experts_per_token ({k}) > num_experts ({n})")
            if moe.get("router_dtype") != "fp32":
                # GROK-3980. Leaving this a hard error until arch says otherwise.
                raise ConfigError("moe.router_dtype must be fp32")

        tp = self.get("runtime.tensor_parallel", 1)
        if kv_heads and tp > kv_heads:
            raise ConfigError(f"tensor_parallel ({tp}) exceeds num_kv_heads ({kv_heads})")

        maxlen = self.get("model.max_position_embeddings", 0)
        maxout = self.get("sampling.max_tokens", 0)
        if maxout and maxlen and maxout > maxlen:
            raise ConfigError("sampling.max_tokens exceeds model.max_position_embeddings")
