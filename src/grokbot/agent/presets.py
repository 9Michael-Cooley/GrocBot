"""Bot presets.

A preset bundles the four things that always get tuned together — persona,
sampling parameters, tool access, and memory policy — under one name. Before
this existed, every caller assembled them independently and drifted; the
`code` surface in one product had temperature 0.7 for six weeks because nobody
noticed it was inheriting the chat default.

Presets are data. Adding one is a dict entry, not a code path.

    from grokbot.agent import PRESETS, get_preset

    preset = get_preset("code")
    completion = engine.generate(prompt, preset.generation_config())

Callers may override any field per request; the preset is the starting point,
not a lock:

    preset.generation_config(temperature=0.0, max_tokens=128)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inference.sampler import GenerationConfig
from ..tokenizer.special import DEFAULT_STOPS
from .persona import Persona, build_system_prompt
from .persona import get as get_persona


@dataclass(frozen=True)
class BotPreset:
    name: str
    description: str
    persona: str = "default"

    # sampling
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    max_tokens: int = 1024

    # agent behaviour
    tools: tuple[str, ...] = ()
    max_iterations: int = 8
    max_tool_calls: int = 24

    # memory
    memory_tokens: int = 32768
    memory_strategy: str = "priority"

    # surfacing
    hidden: bool = False          # not offered in pickers; still selectable by name
    stop: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_STOPS))

    # -- derived ------------------------------------------------------------

    def generation_config(self, **overrides) -> GenerationConfig:
        """Sampling config for this preset. Keyword args win."""
        base = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop=list(self.stop),
        )
        return base.merged(**overrides) if overrides else base

    def get_persona(self) -> Persona:
        return get_persona(self.persona)

    def system_prompt(self, *, tool_schemas: list[dict] | None = None, extra: str = "") -> str:
        return build_system_prompt(self.get_persona(), tool_schemas=tool_schemas, extra=extra)

    def build_agent(self, engine, **kwargs):
        """Construct an Agent wired to this preset.

        Imported lazily — agent.loop imports this module's siblings, and a
        top-level import here closes the cycle.
        """
        from ..tools.registry import REGISTRY
        from .loop import Agent

        registry = kwargs.pop("tools", None)
        if registry is None and self.tools:
            registry = REGISTRY.subset([t for t in self.tools if t in REGISTRY])

        params = dict(
            persona=self.get_persona(),
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            memory_tokens=self.memory_tokens,
        )
        params.update(kwargs)
        return Agent(engine, tools=registry, **params)

    def summary(self) -> str:
        tools = ", ".join(self.tools) if self.tools else "none"
        return (
            f"{self.name:<10} {self.description}\n"
            f"{'':<10} persona={self.persona} temp={self.temperature} "
            f"top_p={self.top_p} max_tokens={self.max_tokens} tools={tools}"
        )


# --------------------------------------------------------------------------
# the presets
# --------------------------------------------------------------------------

DEFAULT = BotPreset(
    name="default",
    description="Balanced general assistant.",
    persona="default",
    tools=("calculator", "clock"),
)

FAST = BotPreset(
    name="fast",
    description="Low latency, short answers. Sized for autocomplete-adjacent UX.",
    persona="concise",
    temperature=0.4,
    top_p=0.9,
    max_tokens=256,
    memory_tokens=8192,
    max_iterations=3,
    tools=(),
)

DEEP = BotPreset(
    name="deep",
    description="Long-form reasoning. Slow and expensive; use it deliberately.",
    persona="research",
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192,
    memory_tokens=131072,
    max_iterations=16,
    max_tool_calls=48,
    tools=("calculator", "clock"),
)

CODE = BotPreset(
    name="code",
    description="Software work. Low temperature; higher invents APIs.",
    persona="code",
    temperature=0.25,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,     # penalising repetition mangles boilerplate
    max_tokens=4096,
    tools=("calculator",),
)

RESEARCH = BotPreset(
    name="research",
    description="Tool-using research assistant. Cites sources, flags uncertainty.",
    persona="research",
    temperature=0.5,
    max_tokens=4096,
    memory_tokens=65536,
    max_iterations=12,
    max_tool_calls=32,
    tools=("calculator", "clock", "doc_search"),
)

CREATIVE = BotPreset(
    name="creative",
    description="Open-ended writing. High temperature, repetition penalty on.",
    persona="default",
    temperature=1.0,
    top_p=0.98,
    top_k=0,                    # 0 disables top-k; nucleus does the work
    min_p=0.02,
    repetition_penalty=1.12,
    max_tokens=4096,
)

FUN = BotPreset(
    name="fun",
    description="Loosened register - jokier, more opinionated. Same safety policy.",
    persona="default",
    temperature=0.95,
    top_p=0.97,
    repetition_penalty=1.08,
    max_tokens=2048,
    tools=("calculator", "clock"),
)

# Used by the eval harness: greedy, no tools, no personality, so runs are
# comparable across checkpoints. Hidden because it produces flat, joyless text
# and someone will otherwise ship it by accident.
EVAL = BotPreset(
    name="eval",
    description="Deterministic greedy decoding for benchmarking.",
    persona="concise",
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    repetition_penalty=1.0,
    max_tokens=2048,
    memory_strategy="truncate",
    hidden=True,
)

PRESETS: dict[str, BotPreset] = {
    p.name: p
    for p in (DEFAULT, FAST, DEEP, CODE, RESEARCH, CREATIVE, FUN, EVAL)
}


def get_preset(name: str) -> BotPreset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; available: {', '.join(list_presets())}"
        ) from None


def list_presets(*, include_hidden: bool = False) -> list[str]:
    return sorted(p.name for p in PRESETS.values() if include_hidden or not p.hidden)


def register_preset(preset: BotPreset, *, replace: bool = False) -> BotPreset:
    if preset.name in PRESETS and not replace:
        raise ValueError(f"preset {preset.name!r} already exists; pass replace=True")
    PRESETS[preset.name] = preset
    return preset


def describe_all(*, include_hidden: bool = False) -> str:
    return "\n\n".join(
        PRESETS[n].summary() for n in list_presets(include_hidden=include_hidden)
    )
