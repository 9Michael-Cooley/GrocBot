"""Streaming output assembly.

Two things are harder than they look and both are handled here.

1. Stop strings can straddle token boundaries. If "END" arrives as "E" + "ND"
   we must not emit the "E" — once it's on the wire we can't take it back. So
   text that could still become a stop string is held until it's resolved.

2. Multi-byte characters straddle token boundaries too. That's the tokenizer's
   decode_incremental problem, not ours; we just never assume one token is one
   renderable string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FinishReason(str, Enum):
    LENGTH = "length"
    STOP = "stop"
    STOP_TOKEN = "stop_token"
    TOOL_CALL = "tool_call"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class StreamChunk:
    text: str = ""
    token_id: int | None = None
    logprob: float | None = None
    index: int = 0
    finish_reason: FinishReason | None = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


@dataclass
class Completion:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    reasoning: str = ""
    finish_reason: FinishReason | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


class StopChecker:
    """Buffers output until stop strings are resolved.

    `feed` returns the text that is safe to emit now. Anything that could still
    turn into a stop string stays in the buffer. Call `flush` at end of stream to
    release whatever was being held.
    """

    def __init__(self, stop_strings: list[str]):
        # Empty stops would match everywhere and pin the buffer forever.
        self.stops = [s for s in stop_strings if s]
        self.max_len = max((len(s) for s in self.stops), default=0)
        self._buffer = ""
        self.triggered: str | None = None

    def _longest_partial_suffix(self, text: str) -> int:
        """Length of the longest suffix of `text` that prefixes some stop string."""
        limit = min(len(text), self.max_len - 1) if self.max_len else 0
        for size in range(limit, 0, -1):
            tail = text[-size:]
            if any(s.startswith(tail) for s in self.stops):
                return size
        return 0

    def feed(self, text: str) -> str:
        if not self.stops:
            return text
        if self.triggered is not None:
            return ""

        self._buffer += text

        earliest = -1
        matched: str | None = None
        for stop in self.stops:
            pos = self._buffer.find(stop)
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest, matched = pos, stop

        if matched is not None:
            self.triggered = matched
            out = self._buffer[:earliest]
            self._buffer = ""
            return out

        hold = self._longest_partial_suffix(self._buffer)
        if hold:
            out, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
        else:
            out, self._buffer = self._buffer, ""
        return out

    def flush(self) -> str:
        out, self._buffer = self._buffer, ""
        return out

    @property
    def stopped(self) -> bool:
        return self.triggered is not None


class StreamAssembler:
    """Per-sequence streaming state: token ids in, StreamChunks out."""

    def __init__(self, tokenizer, cfg, index: int = 0):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.index = index
        self.token_ids: list[int] = []
        self._decoded_upto = 0
        self.stop_checker = StopChecker(list(cfg.stop))
        self.text = ""

    def push(self, token_id: int, logprob: float | None = None) -> StreamChunk | None:
        self.token_ids.append(token_id)
        piece = self.tokenizer.decode_incremental(self.token_ids, self._decoded_upto)
        self._decoded_upto = len(self.token_ids)

        if not piece:
            return None  # mid-character, nothing renderable yet

        emit = self.stop_checker.feed(piece)
        self.text += emit
        if not emit and not self.stop_checker.stopped:
            return None  # held pending a possible stop string

        return StreamChunk(
            text=emit,
            token_id=token_id,
            logprob=logprob,
            index=self.index,
            finish_reason=FinishReason.STOP if self.stop_checker.stopped else None,
        )

    def finalize(self, reason: FinishReason) -> StreamChunk:
        tail = self.stop_checker.flush() if not self.stop_checker.stopped else ""
        self.text += tail
        return StreamChunk(text=tail, index=self.index, finish_reason=reason)

    def completion(self, reason: FinishReason, prompt_tokens: int, cached: int = 0) -> Completion:
        from ..tokenizer.special import strip_thinking

        visible, reasoning = strip_thinking(self.text)
        return Completion(
            text=visible if not self.cfg.echo_reasoning else self.text,
            token_ids=list(self.token_ids),
            reasoning=reasoning,
            finish_reason=reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(self.token_ids),
            cached_tokens=cached,
        )
