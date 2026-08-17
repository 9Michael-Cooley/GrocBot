"""Exception hierarchy.

Everything raised out of grokbot derives from GrokBotError so the serving layer
can map to status codes in one place (see serve/api.py::_status_for).
"""

from __future__ import annotations


class GrokBotError(Exception):
    """Base for everything in this package."""

    status_code = 500


class ConfigError(GrokBotError):
    """Malformed or missing configuration."""

    status_code = 500


class WeightsNotFound(ConfigError):
    """No checkpoint at the configured path.

    Not fatal — the loader falls back to SyntheticBackend and logs loudly.
    Raised only when the caller passed strict=True.
    """


class TokenizerError(GrokBotError):
    status_code = 500


class ContextLengthExceeded(GrokBotError):
    """Prompt + max_tokens exceeds the model's position budget."""

    status_code = 400


class OutOfCacheBlocks(GrokBotError):
    """Allocator could not satisfy a request and preemption did not help.

    The scheduler catches this and requeues; it should never escape to a
    caller. If you see it in a traceback, that's a bug.
    """

    status_code = 503


class SequenceNotFound(GrokBotError):
    status_code = 500


class ToolError(GrokBotError):
    """A tool raised, or was called with arguments that don't match its schema."""

    status_code = 400


class ToolTimeout(ToolError):
    status_code = 504


class SandboxViolation(ToolError):
    """Tool attempted something the sandbox policy denies."""

    status_code = 403


class SafetyBlocked(GrokBotError):
    """Input or output tripped a safety filter."""

    status_code = 400

    def __init__(self, message: str, *, filter_name: str = "", stage: str = ""):
        super().__init__(message)
        self.filter_name = filter_name
        self.stage = stage


class RateLimited(GrokBotError):
    status_code = 429

    def __init__(self, message: str, *, retry_after: float = 1.0):
        super().__init__(message)
        self.retry_after = retry_after


class RequestCancelled(GrokBotError):
    """Client disconnected or abort() was called. Not an error condition."""

    status_code = 499
