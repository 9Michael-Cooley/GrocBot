"""Metrics.

Prometheus text exposition, no client library — the monorepo forbade the
dependency and the format is twenty lines to emit correctly.

Histogram buckets are explicit and deliberately dense in the 20-500ms band. The
default client buckets put four boundaries above 1s and two below 100ms, which
is exactly backwards for TTFT: everything lands in one bucket and the p99 is
interpolated out of nothing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0, 2.0, 5.0, 10.0)
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
TOKEN_BUCKETS = (16, 64, 128, 512, 1024, 2048, 8192, 32768, 131072)


@dataclass
class Counter:
    name: str
    help: str = ""
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counters cannot decrease")
        self.value += amount


@dataclass
class Gauge:
    name: str
    help: str = ""
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class Histogram:
    name: str
    help: str = ""
    buckets: tuple[float, ...] = LATENCY_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)   # +1 for +Inf

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def quantile(self, q: float) -> float:
        """Interpolated from buckets, so it inherits their resolution. Reported
        as an estimate because it is one."""
        if not self.count:
            return 0.0
        target = q * self.count
        cumulative = 0
        for i, edge in enumerate(self.buckets):
            cumulative += self.counts[i]
            if cumulative >= target:
                lower = self.buckets[i - 1] if i else 0.0
                span = self.counts[i] or 1
                prior = cumulative - self.counts[i]
                return lower + (edge - lower) * ((target - prior) / span)
        return self.buckets[-1]

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class Registry:
    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str = "") -> Counter:
        return self._get_or_create(name, Counter(name, help))

    def gauge(self, name: str, help: str = "") -> Gauge:
        return self._get_or_create(name, Gauge(name, help))

    def histogram(self, name: str, help: str = "", buckets=LATENCY_BUCKETS) -> Histogram:
        return self._get_or_create(name, Histogram(name, help, buckets))

    def _get_or_create(self, name, default):
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = default
            existing = self._metrics[name]
        if type(existing) is not type(default):
            raise TypeError(
                f"metric {name!r} already registered as {type(existing).__name__}"
            )
        return existing

    def render(self) -> str:
        """Prometheus text format v0.0.4."""
        lines: list[str] = []
        with self._lock:
            metrics = list(self._metrics.values())

        for m in metrics:
            kind = {"Counter": "counter", "Gauge": "gauge", "Histogram": "histogram"}[
                type(m).__name__
            ]
            if m.help:
                lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} {kind}")

            if isinstance(m, Histogram):
                cumulative = 0
                for i, edge in enumerate(m.buckets):
                    cumulative += m.counts[i]
                    lines.append(f'{m.name}_bucket{{le="{edge}"}} {cumulative}')
                cumulative += m.counts[-1]
                lines.append(f'{m.name}_bucket{{le="+Inf"}} {cumulative}')
                lines.append(f"{m.name}_sum {m.total}")
                lines.append(f"{m.name}_count {m.count}")
            else:
                lines.append(f"{m.name} {m.value}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


REGISTRY = Registry()

requests_total = REGISTRY.counter("grokbot_requests_total", "Requests received.")
requests_failed = REGISTRY.counter("grokbot_requests_failed_total", "Requests ending in error.")
tokens_generated = REGISTRY.counter("grokbot_tokens_generated_total", "Output tokens produced.")
tokens_prefilled = REGISTRY.counter("grokbot_tokens_prefilled_total", "Prompt tokens processed.")
preemptions = REGISTRY.counter("grokbot_preemptions_total", "Sequences preempted (GROK-4417).")
cache_hits = REGISTRY.counter("grokbot_prefix_cache_hits_total", "Prefix cache block hits.")

running_sequences = REGISTRY.gauge("grokbot_running_sequences", "Sequences in the running batch.")
waiting_sequences = REGISTRY.gauge("grokbot_waiting_sequences", "Sequences queued.")
cache_utilization = REGISTRY.gauge("grokbot_kv_cache_utilization", "Fraction of blocks in use.")

ttft = REGISTRY.histogram("grokbot_ttft_seconds", "Time to first token.", TTFT_BUCKETS)
latency = REGISTRY.histogram("grokbot_request_duration_seconds", "End-to-end request duration.")
prompt_tokens = REGISTRY.histogram("grokbot_prompt_tokens", "Prompt length.", TOKEN_BUCKETS)


class Timer:
    """`with Timer(hist):` — observes on exit, including on exception."""

    def __init__(self, histogram: Histogram):
        self.histogram = histogram
        self.start = 0.0

    def __enter__(self) -> Timer:
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        self.histogram.observe(time.monotonic() - self.start)


def scrape_engine(engine) -> None:
    """Pull scheduler/cache gauges. Called on /metrics rather than continuously —
    these are cheap to read and there is no point sampling them on a timer."""
    snap = engine.scheduler.snapshot()
    running_sequences.set(snap["running"])
    waiting_sequences.set(snap["waiting"])
    cache_utilization.set(snap["cache"]["utilization"])
