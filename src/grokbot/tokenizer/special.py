"""Special tokens and chat templating.

The control-token set is frozen — ids are baked into the checkpoints. Adding a
token means a retrain, not a config change. Order in SPECIAL_TOKENS defines id
assignment for the synthetic vocab only.
"""

from __future__ import annotations

BOS = "<|bos|>"
EOS = "<|eot|>"
PAD = "<|pad|>"
UNK = "<|unk|>"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "<|/tool_call|>"
TOOL_RESULT_START = "<|tool_result|>"
TOOL_RESULT_END = "<|/tool_result|>"

THINK_START = "<|think|>"
THINK_END = "<|/think|>"

SPECIAL_TOKENS = [
    BOS,
    EOS,
    PAD,
    UNK,
    IM_START,
    IM_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESULT_START,
    TOOL_RESULT_END,
    THINK_START,
    THINK_END,
]

# Tokens the sampler must never emit. PAD/UNK showing up in output has meant a
# corrupt checkpoint every time it's happened.
SUPPRESSED = [PAD, UNK, BOS]

# Default stops. Callers can extend but not remove IM_END.
DEFAULT_STOPS = [IM_END, EOS]

VALID_ROLES = ("system", "user", "assistant", "tool")


def render_chat(messages: list[dict], *, add_generation_prompt: bool = True) -> str:
    """ChatML. Tool results are folded in as their own turn.

    Kept as string formatting rather than a template engine on purpose — this
    runs per request and the template never varies per deployment.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"unknown role {role!r}, expected one of {VALID_ROLES}")
        content = msg.get("content") or ""

        if role == "tool":
            name = msg.get("name", "unknown")
            parts.append(
                f"{IM_START}tool\n{TOOL_RESULT_START}{name}\n{content}\n{TOOL_RESULT_END}{IM_END}\n"
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            import json

            calls = "\n".join(
                f"{TOOL_CALL_START}{json.dumps(c, separators=(',', ':'))}{TOOL_CALL_END}"
                for c in msg["tool_calls"]
            )
            body = f"{content}\n{calls}" if content else calls
            parts.append(f"{IM_START}assistant\n{body}{IM_END}\n")
            continue

        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def strip_thinking(text: str) -> tuple[str, str]:
    """Split a completion into (visible, reasoning).

    Reasoning blocks are not returned to API callers unless the request opted in.
    Unterminated blocks (hit max_tokens mid-thought) count as entirely reasoning.
    """
    if THINK_START not in text:
        return text, ""
    head, _, rest = text.partition(THINK_START)
    thought, sep, tail = rest.partition(THINK_END)
    if not sep:
        return head, thought
    return head + tail, thought
