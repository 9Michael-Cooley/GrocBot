"""The agent loop.

    render memory -> generate -> parse tool calls -> execute -> record -> repeat

Terminates when the model stops emitting tool calls, when max_iterations is hit,
or when a repeated-failure guard trips.

The guards are the whole point. An agent loop without them is a machine for
turning a transient tool error into an unbounded bill: the model retries the
same failing call, gets the same error, and has no mechanism to notice. Three
guards run here:

  budget      hard ceiling on iterations and total tool calls
  repetition  identical (tool, args) twice in a row is refused
  failure     N consecutive tool errors ends the loop
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..errors import SafetyBlocked, ToolError
from ..inference.sampler import GenerationConfig
from ..safety.policy import PolicyEngine
from ..tokenizer.special import TOOL_CALL_END, TOOL_CALL_START
from ..tools.registry import REGISTRY, ToolRegistry
from ..tools.sandbox import Sandbox, SandboxPolicy
from ..utils.logging import get_logger
from .memory import WorkingMemory
from .persona import DEFAULT, Persona, build_system_prompt

log = get_logger(__name__)

_CALL_RE = re.compile(
    re.escape(TOOL_CALL_START) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_END),
    re.DOTALL,
)
# Models drop the closing token when they hit max_tokens. Salvage that case.
_UNTERMINATED_RE = re.compile(re.escape(TOOL_CALL_START) + r"\s*(\{.*)", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""

    def key(self) -> str:
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True)}"


@dataclass
class Step:
    iteration: int
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class AgentResult:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    stopped_because: str = "completed"
    total_tool_calls: int = 0
    duration_s: float = 0.0

    @property
    def iterations(self) -> int:
        return len(self.steps)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split a completion into (prose, calls). Malformed JSON is skipped, not
    raised — one bad call should not lose a turn that also contains good ones."""
    calls: list[ToolCall] = []
    for match in _CALL_RE.finditer(text):
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError as exc:
            log.warning("unparseable tool call %r: %s", blob[:120], exc)
            continue
        name = payload.get("name")
        if not name:
            log.warning("tool call missing 'name': %r", blob[:120])
            continue
        args = payload.get("arguments", payload.get("parameters", {}))
        calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}, raw=blob))

    prose = _CALL_RE.sub("", text)

    if not calls:
        trailing = _UNTERMINATED_RE.search(prose)
        if trailing:
            try:
                payload = json.loads(trailing.group(1))
                if payload.get("name"):
                    calls.append(
                        ToolCall(payload["name"], payload.get("arguments", {}), trailing.group(1))
                    )
                    prose = prose[: trailing.start()]
            except json.JSONDecodeError:
                pass   # genuinely truncated; nothing to recover

    return prose.strip(), calls


class Agent:
    def __init__(
        self,
        engine,
        *,
        tools: ToolRegistry | None = None,
        persona: Persona | str = DEFAULT,
        policy: PolicyEngine | None = None,
        sandbox: Sandbox | None = None,
        max_iterations: int = 8,
        max_tool_calls: int = 24,
        max_consecutive_failures: int = 3,
        memory_tokens: int = 32768,
        system_extra: str = "",
    ):
        self.engine = engine
        self.tools = tools if tools is not None else REGISTRY
        self.policy = policy or PolicyEngine()
        self.sandbox = sandbox or Sandbox(SandboxPolicy())
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_consecutive_failures = max_consecutive_failures

        system = build_system_prompt(
            persona, tool_schemas=self.tools.schemas(), extra=system_extra
        )
        self.memory = WorkingMemory(
            engine.tokenizer,
            max_tokens=memory_tokens,
            system_prompt=system,
        )

    # -- execution ---------------------------------------------------------

    def _execute(self, call: ToolCall) -> dict:
        decision = self.policy.check_tool_args(call.name, call.arguments)
        if not decision.allowed:
            return {"tool": call.name, "ok": False, "error": f"blocked: {decision.reason}"}

        try:
            tool = self.tools.get(call.name)
        except ToolError as exc:
            return {"tool": call.name, "ok": False, "error": str(exc)}

        if tool.dangerous:
            return {
                "tool": call.name,
                "ok": False,
                "error": f"{call.name} is marked dangerous and is not enabled in this registry",
            }

        try:
            args = tool.validate_args(call.arguments)
        except ToolError as exc:
            return {"tool": call.name, "ok": False, "error": str(exc)}

        try:
            outcome = self.sandbox.run(tool.fn, args)
        except ToolError as exc:
            return {"tool": call.name, "ok": False, "error": str(exc)}

        rendered = str(outcome.value)
        checked = self.policy.check_tool_result(call.name, rendered)
        if not checked.allowed:
            return {"tool": call.name, "ok": False, "error": f"result blocked: {checked.reason}"}

        return {
            "tool": call.name,
            "ok": True,
            "result": checked.text,
            "duration_s": round(outcome.duration_s, 4),
        }

    # -- driver ------------------------------------------------------------

    def run(self, prompt: str, gen: GenerationConfig | None = None) -> AgentResult:
        started = time.monotonic()
        result = AgentResult()

        decision = self.policy.check_input(prompt)
        if not decision.allowed:
            raise SafetyBlocked(decision.reason, stage="input")

        self.memory.add("user", decision.text)

        last_key: str | None = None
        consecutive_failures = 0

        for iteration in range(1, self.max_iterations + 1):
            step_start = time.monotonic()
            completion = self.engine.chat(self.memory.messages(), gen)
            prose, calls = parse_tool_calls(completion.text)
            step = Step(iteration=iteration, text=prose, tool_calls=calls)

            if not calls:
                out = self.policy.check_output(prose)
                result.answer = out.text if out.allowed else _refusal()
                self.memory.add("assistant", result.answer)
                step.duration_s = time.monotonic() - step_start
                result.steps.append(step)
                break

            self.memory.add(
                "assistant",
                prose,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in calls],
            )

            for call in calls:
                if result.total_tool_calls >= self.max_tool_calls:
                    result.stopped_because = "tool call budget exhausted"
                    break

                if call.key() == last_key:
                    outcome = {
                        "tool": call.name,
                        "ok": False,
                        "error": (
                            "identical call was just made and returned the same thing. "
                            "Use the previous result or try different arguments."
                        ),
                    }
                else:
                    outcome = self._execute(call)
                last_key = call.key()

                result.total_tool_calls += 1
                step.results.append(outcome)
                self.memory.add_tool_result(
                    call.name,
                    outcome.get("result") if outcome["ok"] else f"ERROR: {outcome['error']}",
                )

                consecutive_failures = 0 if outcome["ok"] else consecutive_failures + 1

            step.duration_s = time.monotonic() - step_start
            result.steps.append(step)

            if consecutive_failures >= self.max_consecutive_failures:
                result.stopped_because = f"{consecutive_failures} consecutive tool failures"
                result.answer = (
                    "I stopped because the tools I need keep failing. The last error was: "
                    + str(step.results[-1].get("error", "unknown"))
                )
                break
            if result.stopped_because != "completed":
                break
        else:
            result.stopped_because = f"hit max_iterations ({self.max_iterations})"
            result.answer = result.answer or "I ran out of steps before finishing."

        result.duration_s = time.monotonic() - started
        return result

    def stream(self, prompt: str, gen: GenerationConfig | None = None) -> Iterator[Step]:
        """Same loop, yielding each step as it completes. The final step holds
        the answer."""
        outcome = self.run(prompt, gen)
        yield from outcome.steps

    def reset(self) -> None:
        self.memory.clear()


def _refusal() -> str:
    from ..safety.policy import REFUSAL

    return REFUSAL
