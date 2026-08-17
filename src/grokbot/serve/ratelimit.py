"""Rate limiting.

Token bucket per key, on two dimensions (requests and tokens) plus a concurrency
semaphore. Token bucket rather than a fixed window because a fixed window lets a
client spend its entire minute's budget in the last second and then the next
minute's in the first — a 2x burst straddling the boundary.

KNOWN ISSUE: state is per-process. With server.workers > 1 or more than one
replica, the effective limit is N times what is configured. Everything real runs
behind an edge limiter, so this is defence in depth rather than the control, but
it is still wrong. Moving it to shared state needs Redis, which the serving pod
does not currently have.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..errors import RateLimited


@dataclass
class Bucket:
    capacity: float
    refill_per_second: float
    tokens: float = 0.0
    last: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = self.capacity

    def _refill(self, now: float) -> None:
        elapsed = now - self.last
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.last = now

    def try_consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float = 1.0) -> float:
        deficit = amount - self.tokens
        if deficit <= 0:
            return 0.0
        return deficit / self.refill_per_second if self.refill_per_second else float("inf")


@dataclass
class LimitConfig:
    requests_per_minute: int = 600
    tokens_per_minute: int = 250_000
    concurrent_per_key: int = 16
    max_prompt_tokens: int = 120_000
    max_output_tokens: int = 8192
    burst_multiplier: float = 1.5

    @classmethod
    def from_dict(cls, data: dict | None) -> LimitConfig:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})


class RateLimiter:
    def __init__(self, cfg: LimitConfig | None = None):
        self.cfg = cfg or LimitConfig()
        self._requests: dict[str, Bucket] = {}
        self._tokens: dict[str, Bucket] = {}
        self._concurrent: dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}

    def _buckets(self, key: str) -> tuple[Bucket, Bucket]:
        cfg = self.cfg
        if key not in self._requests:
            self._requests[key] = Bucket(
                capacity=cfg.requests_per_minute * cfg.burst_multiplier,
                refill_per_second=cfg.requests_per_minute / 60.0,
            )
            self._tokens[key] = Bucket(
                capacity=cfg.tokens_per_minute * cfg.burst_multiplier,
                refill_per_second=cfg.tokens_per_minute / 60.0,
            )
        self._last_seen[key] = time.monotonic()
        return self._requests[key], self._tokens[key]

    def check(self, key: str, estimated_tokens: int = 0) -> None:
        """Raises RateLimited, or returns and has consumed the budget."""
        with self._lock:
            req_bucket, tok_bucket = self._buckets(key)

            if self._concurrent.get(key, 0) >= self.cfg.concurrent_per_key:
                raise RateLimited(
                    f"at most {self.cfg.concurrent_per_key} concurrent requests per key",
                    retry_after=1.0,
                )
            if not req_bucket.try_consume(1.0):
                raise RateLimited(
                    f"request rate limit ({self.cfg.requests_per_minute}/min) exceeded",
                    retry_after=req_bucket.retry_after(1.0),
                )
            if estimated_tokens and not tok_bucket.try_consume(estimated_tokens):
                # Give back the request token — the request never ran.
                req_bucket.tokens += 1.0
                raise RateLimited(
                    f"token rate limit ({self.cfg.tokens_per_minute}/min) exceeded",
                    retry_after=tok_bucket.retry_after(estimated_tokens),
                )

    def acquire(self, key: str) -> None:
        with self._lock:
            self._concurrent[key] = self._concurrent.get(key, 0) + 1

    def release(self, key: str) -> None:
        with self._lock:
            remaining = self._concurrent.get(key, 0) - 1
            if remaining > 0:
                self._concurrent[key] = remaining
            else:
                self._concurrent.pop(key, None)

    def validate_size(self, prompt_tokens: int, max_output: int) -> None:
        if prompt_tokens > self.cfg.max_prompt_tokens:
            raise RateLimited(
                f"prompt is {prompt_tokens} tokens, limit is {self.cfg.max_prompt_tokens}",
                retry_after=0.0,
            )
        if max_output > self.cfg.max_output_tokens:
            raise RateLimited(
                f"max_tokens {max_output} exceeds the limit of {self.cfg.max_output_tokens}",
                retry_after=0.0,
            )

    def gc(self, older_than_s: float = 3600.0) -> int:
        """Drop idle keys. Without this the dicts grow forever on deployments
        with per-user keys."""
        cutoff = time.monotonic() - older_than_s
        with self._lock:
            stale = [k for k, ts in self._last_seen.items() if ts < cutoff]
            for k in stale:
                self._requests.pop(k, None)
                self._tokens.pop(k, None)
                self._last_seen.pop(k, None)
        return len(stale)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "tracked_keys": len(self._requests),
                "active_concurrent": sum(self._concurrent.values()),
            }


class Guard:
    """`with Guard(limiter, key):` — releases the concurrency slot on exit."""

    def __init__(self, limiter: RateLimiter, key: str):
        self.limiter = limiter
        self.key = key

    def __enter__(self) -> Guard:
        self.limiter.acquire(self.key)
        return self

    def __exit__(self, *exc) -> None:
        self.limiter.release(self.key)
