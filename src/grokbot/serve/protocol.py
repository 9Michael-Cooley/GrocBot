"""OpenAI-compatible request/response translation.

Compatibility is the point: every client library already speaks this shape. It
means inheriting some decisions we would not have made (`max_tokens` vs
`max_completion_tokens`, top-level `stop` accepting either a string or a list),
which is why the coercion is all in one place instead of scattered through the
handler.

Unsupported-but-accepted fields are listed in IGNORED. They are accepted and
dropped rather than rejected, because clients send them by default and a 400 on
`user` or `logprobs` breaks integrations for no benefit. They are counted so the
gap is at least visible.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from ..errors import GrokBotError
from ..inference.sampler import GenerationConfig
from ..inference.stream import Completion, FinishReason

IGNORED = {
    "user", "logprobs", "top_logprobs", "logit_bias_type", "service_tier",
    "parallel_tool_calls", "response_format", "seed_mode", "store", "metadata",
}

_ignored_counts: dict[str, int] = {}

_FINISH_MAP = {
    FinishReason.LENGTH: "length",
    FinishReason.STOP: "stop",
    FinishReason.STOP_TOKEN: "stop",
    FinishReason.TOOL_CALL: "tool_calls",
    FinishReason.CANCELLED: "stop",
    FinishReason.ERROR: "stop",
}


class ProtocolError(GrokBotError):
    status_code = 400


@dataclass
class ChatRequest:
    messages: list[dict]
    model: str = "grok-3-mini"
    stream: bool = False
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    tools: list[dict] = field(default_factory=list)
    tool_choice: str = "auto"

    @classmethod
    def parse(cls, payload: dict) -> ChatRequest:
        if not isinstance(payload, dict):
            raise ProtocolError("request body must be a JSON object")

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ProtocolError("'messages' must be a non-empty array")

        normalized: list[dict] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or "role" not in msg:
                raise ProtocolError(f"messages[{i}] must be an object with a 'role'")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal content blocks. Text parts only — this build has no
                # vision tower, and silently dropping an image would produce a
                # confidently wrong answer about a picture nobody looked at.
                for part in content:
                    if isinstance(part, dict) and part.get("type") not in ("text", None):
                        raise ProtocolError(
                            f"messages[{i}]: content type {part.get('type')!r} is not supported"
                        )
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            normalized.append({**msg, "content": content or ""})

        for key in payload:
            if key in IGNORED:
                _ignored_counts[key] = _ignored_counts.get(key, 0) + 1

        stop = payload.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if len(stop) > 8:
            raise ProtocolError(f"at most 8 stop sequences, got {len(stop)}")

        bias = payload.get("logit_bias") or {}
        try:
            bias = {int(k): float(v) for k, v in bias.items()}
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"logit_bias must map token ids to numbers: {exc}") from exc

        gen = GenerationConfig(
            max_tokens=int(
                payload.get("max_completion_tokens") or payload.get("max_tokens") or 512
            ),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.95)),
            top_k=int(payload.get("top_k", 64)),
            frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
            presence_penalty=float(payload.get("presence_penalty", 0.0)),
            stop=stop,
            logit_bias=bias,
            seed=payload.get("seed"),
            n=int(payload.get("n", 1)),
        )
        if gen.n != 1:
            raise ProtocolError("n > 1 is not supported; issue parallel requests instead")

        return cls(
            messages=normalized,
            model=payload.get("model", "grok-3-mini"),
            stream=bool(payload.get("stream", False)),
            gen=gen,
            tools=payload.get("tools") or [],
            tool_choice=payload.get("tool_choice", "auto"),
        )


def _request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def chat_response(completion: Completion, model: str, request_id: str | None = None) -> dict:
    return {
        "id": request_id or _request_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": _FINISH_MAP.get(completion.finish_reason, "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "prompt_tokens_details": {"cached_tokens": completion.cached_tokens},
        },
    }


def chunk_response(
    text: str, model: str, request_id: str, *, role: bool = False, finish: str | None = None
) -> dict:
    delta: dict = {}
    if role:
        delta["role"] = "assistant"
    if text:
        delta["content"] = text
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def sse(payload: dict | str) -> bytes:
    """Server-sent event frame. The blank line terminator is not optional —
    without it clients buffer until the connection closes, which looks exactly
    like the model being slow."""
    body = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {body}\n\n".encode()


SSE_DONE = b"data: [DONE]\n\n"


def error_response(exc: Exception, status: int = 500) -> dict:
    kind = type(exc).__name__
    return {
        "error": {
            "message": str(exc),
            "type": kind,
            "code": getattr(exc, "status_code", status),
        }
    }


def ignored_field_report() -> dict[str, int]:
    return dict(_ignored_counts)
