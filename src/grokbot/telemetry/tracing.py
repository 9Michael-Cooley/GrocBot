"""Tracing.

Minimal span tree with an OTLP-shaped export. The collector endpoint defaulted
to the internal one; it is None here and the exporter is a no-op unless
configured. Turning it on without an endpoint buffers spans and drops the oldest,
it does not attempt network calls.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger(__name__)

_MAX_BUFFERED_SPANS = 4096


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    events: list[tuple[float, str, dict]] = field(default_factory=list)
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000.0

    def set(self, key: str, value) -> Span:
        self.attributes[key] = value
        return self

    def event(self, name: str, **attrs) -> Span:
        self.events.append((time.time(), name, attrs))
        return self

    def error(self, exc: BaseException) -> Span:
        self.status = "error"
        self.attributes["error.type"] = type(exc).__name__
        self.attributes["error.message"] = str(exc)[:500]
        return self

    def to_otlp(self) -> dict:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_id or "",
            "name": self.name,
            "startTimeUnixNano": int(self.start * 1e9),
            "endTimeUnixNano": int((self.end or time.time()) * 1e9),
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in self.attributes.items()],
            "events": [
                {"timeUnixNano": int(ts * 1e9), "name": n, "attributes": a}
                for ts, n, a in self.events
            ],
            "status": {"code": 2 if self.status == "error" else 1},
        }


class Tracer:
    def __init__(self, enabled: bool = False, endpoint: str | None = None, service: str = "grokbot"):
        self.enabled = enabled
        self.endpoint = endpoint or os.environ.get("GROKBOT_OTLP_ENDPOINT")
        self.service = service
        self._local = threading.local()
        self._finished: deque[Span] = deque(maxlen=_MAX_BUFFERED_SPANS)

        if self.enabled and not self.endpoint:
            log.warning("tracing enabled with no OTLP endpoint; spans buffer in memory only")

    @property
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @contextlib.contextmanager
    def span(self, name: str, **attributes):
        if not self.enabled:
            yield _NULL_SPAN
            return

        parent = self._stack[-1] if self._stack else None
        current = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        self._stack.append(current)
        try:
            yield current
        except BaseException as exc:
            current.error(exc)
            raise
        finally:
            current.end = time.time()
            self._stack.pop()
            self._finished.append(current)

    def current_trace_id(self) -> str | None:
        return self._stack[-1].trace_id if self._stack else None

    def export(self) -> str:
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service}}
                        ]
                    },
                    "scopeSpans": [{"spans": [s.to_otlp() for s in self._finished]}],
                }
            ]
        }
        return json.dumps(payload, separators=(",", ":"))

    def drain(self) -> list[Span]:
        spans = list(self._finished)
        self._finished.clear()
        return spans

    def summary(self) -> dict:
        by_name: dict[str, list[float]] = {}
        for s in self._finished:
            by_name.setdefault(s.name, []).append(s.duration_ms)
        return {
            name: {
                "count": len(v),
                "mean_ms": round(sum(v) / len(v), 3),
                "max_ms": round(max(v), 3),
            }
            for name, v in sorted(by_name.items())
        }


class _NullSpan:
    """Zero-cost stand-in when tracing is off."""

    def set(self, *_a, **_k):
        return self

    def event(self, *_a, **_k):
        return self

    def error(self, *_a, **_k):
        return self


_NULL_SPAN = _NullSpan()
TRACER = Tracer(enabled=False)


def configure(enabled: bool, endpoint: str | None = None) -> Tracer:
    global TRACER
    TRACER = Tracer(enabled=enabled, endpoint=endpoint)
    return TRACER
