"""System prompt construction.

Personas are data, not code. They are assembled here rather than pasted into
call sites so that a change to the tool-calling contract is one edit instead of
a grep across the repo.

The tool-protocol section is generated from the registry. Hand-writing it means
the prompt describes tools that don't exist, which the model will then try to
call — this was the top source of malformed tool calls before it was generated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..tokenizer.special import TOOL_CALL_END, TOOL_CALL_START


@dataclass
class Persona:
    name: str = "default"
    instructions: str = ""
    traits: list[str] = field(default_factory=list)
    refusal_style: str = "brief"
    verbosity: str = "balanced"        # terse | balanced | thorough
    allow_reasoning: bool = True

    def render(self) -> str:
        parts = [self.instructions.strip()] if self.instructions.strip() else []
        if self.traits:
            parts.append("Style: " + "; ".join(self.traits) + ".")
        parts.append(_VERBOSITY[self.verbosity])
        if self.refusal_style == "brief":
            parts.append(
                "If you decline something, say so in one sentence and offer the closest "
                "thing you can do. Do not lecture."
            )
        return "\n\n".join(p for p in parts if p)


_VERBOSITY = {
    "terse": "Answer in as few words as the question allows. No preamble, no summary.",
    "balanced": "Match the length of your answer to the complexity of the question.",
    "thorough": "Work through the problem explicitly. Show the reasoning that matters.",
}


DEFAULT = Persona(
    name="default",
    instructions=(
        "You are Grok, a large language model. You are direct, technically precise, and "
        "willing to say when you do not know something."
    ),
    traits=["dry rather than enthusiastic", "concrete over abstract"],
)

CONCISE = Persona(
    name="concise",
    instructions="You are Grok. Answer questions directly.",
    verbosity="terse",
    allow_reasoning=False,
)

RESEARCH = Persona(
    name="research",
    instructions=(
        "You are Grok operating as a research assistant. Cite what you used. Separate "
        "what a source says from what you infer from it. When evidence is thin, say so "
        "rather than hedging with vague language."
    ),
    traits=["skeptical", "explicit about uncertainty"],
    verbosity="thorough",
)

CODE = Persona(
    name="code",
    instructions=(
        "You are Grok assisting with software. Prefer working code over description. "
        "Match the conventions of the surrounding codebase. Point out a bug you notice "
        "even if it is not what was asked about."
    ),
    traits=["precise", "no unnecessary commentary in code"],
)

PERSONAS = {p.name: p for p in (DEFAULT, CONCISE, RESEARCH, CODE)}


def get(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError:
        raise ValueError(f"unknown persona {name!r}; available: {sorted(PERSONAS)}") from None


def tool_protocol(schemas: list[dict]) -> str:
    """The tool-calling contract, generated from live schemas."""
    if not schemas:
        return ""

    lines = [
        "You have access to the following tools.",
        "",
        "To call one, emit exactly:",
        f'{TOOL_CALL_START}{{"name": "<tool>", "arguments": {{...}}}}{TOOL_CALL_END}',
        "",
        "Rules:",
        "- Emit nothing after a tool call in the same turn. Wait for the result.",
        "- Call a tool only when you need information you do not have. Do not call one "
        "to confirm something you already know.",
        "- Arguments must match the schema exactly. Do not invent parameters.",
        "- If a tool fails twice with the same arguments, stop and explain the failure "
        "instead of retrying.",
        "",
        "Tools:",
    ]
    for s in schemas:
        fn = s["function"]
        params = fn.get("parameters", {})
        lines.append(f"\n{fn['name']}: {fn['description']}")
        lines.append(f"  parameters: {json.dumps(params, separators=(',', ':'))}")
    return "\n".join(lines)


def build_system_prompt(
    persona: Persona | str = DEFAULT,
    *,
    tool_schemas: list[dict] | None = None,
    extra: str = "",
) -> str:
    p = get(persona) if isinstance(persona, str) else persona
    sections = [p.render()]
    if tool_schemas:
        sections.append(tool_protocol(tool_schemas))
    if extra.strip():
        sections.append(extra.strip())
    return "\n\n".join(s for s in sections if s)
