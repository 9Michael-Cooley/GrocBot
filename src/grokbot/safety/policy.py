"""Policy engine.

Wraps the filters into a request-lifecycle decision: what to do at input, at
output, and around tool calls. The classifier that handles anything requiring
judgement runs as a separate service; `PolicyEngine.classify` is the client
interface and there is no client in this tree, so it fails open with a warning.
Fail-open is the right default for a *local* dev tree and the wrong default for
anything else — serving overrides it via `on_classifier_error`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..errors import SafetyBlocked
from ..utils.logging import get_logger
from .filters import Action, Filter, FilterHit, build_filters, run_filters

log = get_logger(__name__)


class Stage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    TOOL_ARGS = "tool_args"
    TOOL_RESULT = "tool_result"


@dataclass
class PolicyDecision:
    allowed: bool
    text: str
    stage: Stage
    hits: list[FilterHit] = field(default_factory=list)
    reason: str = ""

    def raise_if_blocked(self) -> None:
        if not self.allowed:
            hit = self.hits[0] if self.hits else None
            raise SafetyBlocked(
                self.reason or "blocked by policy",
                filter_name=hit.filter_name if hit else "",
                stage=self.stage.value,
            )


REFUSAL = (
    "I can't help with that. If you think this is a mistake, rephrase the request "
    "or contact the operator of this deployment."
)


@dataclass
class PolicyConfig:
    enabled: bool = True
    input_filters: list[str] = field(default_factory=lambda: ["pii", "injection"])
    output_filters: list[str] = field(default_factory=lambda: ["pii", "leakage"])
    on_block: str = "refuse"                 # refuse | redact | error
    on_classifier_error: str = "allow"       # allow | block
    log_blocked: bool = True
    tool_arg_filters: list[str] = field(default_factory=lambda: ["injection"])

    @classmethod
    def from_dict(cls, data: dict | None) -> PolicyConfig:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})


class PolicyEngine:
    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self._input: list[Filter] = build_filters(self.cfg.input_filters)
        self._output: list[Filter] = build_filters(self.cfg.output_filters)
        self._tool_args: list[Filter] = build_filters(self.cfg.tool_arg_filters)
        self._counts: dict[str, int] = {}

    # -- classifier --------------------------------------------------------

    def classify(self, text: str) -> tuple[bool, str]:
        """Call the policy model. Not available here.

        Returns (allowed, reason). The real client batches, caches on a hash of
        the text, and has a 200ms budget after which it fails according to
        on_classifier_error.
        """
        if self.cfg.on_classifier_error == "block":
            return False, "policy classifier unavailable and on_classifier_error=block"
        log.debug("policy classifier unavailable; failing open")
        return True, ""

    # -- stages ------------------------------------------------------------

    def _evaluate(self, text: str, stage: Stage, filters: list[Filter]) -> PolicyDecision:
        if not self.cfg.enabled:
            return PolicyDecision(True, text, stage)

        result = run_filters(text, filters)
        for hit in result.hits:
            key = f"{hit.filter_name}.{hit.pattern_name}"
            self._counts[key] = self._counts.get(key, 0) + 1

        if result.blocked:
            blocking = next(h for h in result.hits if h.action is Action.BLOCK)
            reason = f"{blocking.filter_name}/{blocking.pattern_name} matched at {stage.value}"
            if self.cfg.log_blocked:
                # Excerpt only — never the full text. Blocked prompts are exactly
                # the ones you least want sitting in a log aggregator.
                log.warning("policy block: %s excerpt=%r", reason, blocking.excerpt[:60])
            if self.cfg.on_block == "redact":
                return PolicyDecision(True, REFUSAL, stage, result.hits, reason)
            return PolicyDecision(False, result.text, stage, result.hits, reason)

        return PolicyDecision(True, result.text, stage, result.hits)

    def check_input(self, text: str) -> PolicyDecision:
        decision = self._evaluate(text, Stage.INPUT, self._input)
        if decision.allowed:
            ok, reason = self.classify(text)
            if not ok:
                return PolicyDecision(False, text, Stage.INPUT, decision.hits, reason)
        return decision

    def check_output(self, text: str) -> PolicyDecision:
        return self._evaluate(text, Stage.OUTPUT, self._output)

    def check_tool_args(self, tool_name: str, args: dict) -> PolicyDecision:
        rendered = " ".join(f"{k}={v}" for k, v in args.items())
        decision = self._evaluate(rendered, Stage.TOOL_ARGS, self._tool_args)
        if not decision.allowed:
            decision.reason = f"{tool_name}: {decision.reason}"
        return decision

    def check_tool_result(self, tool_name: str, result: str) -> PolicyDecision:
        """Tool output is untrusted input. It came from the internet, a database,
        or a file — all places an attacker can write to. It gets filtered on the
        way back in, same as a user turn."""
        return self._evaluate(result, Stage.TOOL_RESULT, self._input)

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict:
        return {"enabled": self.cfg.enabled, "hits": dict(self._counts)}

    def reset_stats(self) -> None:
        self._counts.clear()
