"""Working memory for the agent loop.

Holds the conversation and decides what survives when it no longer fits. Three
strategies, in increasing order of how much they cost and how well they work:

  truncate   drop oldest turns. Cheap, and the model forgets the task.
  summarize  fold dropped turns into a running summary via the model itself.
  priority   score turns and keep the important ones regardless of age.

`priority` is the default because the failure mode of the other two is the same:
the agent forgets what it was asked to do around turn 30 and starts improvising.

The system turn is pinned in all strategies. Dropping it is never correct and
the code should not permit it, which is why it isn't part of `turns` at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Turn:
    role: str
    content: str
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    tool_name: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    pinned: bool = False

    def to_message(self) -> dict:
        msg: dict = {"role": self.role, "content": self.content}
        if self.tool_name:
            msg["name"] = self.tool_name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class WorkingMemory:
    def __init__(
        self,
        tokenizer,
        *,
        max_tokens: int = 32768,
        reserve_for_output: int = 4096,
        strategy: str = "priority",
        system_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.reserve = reserve_for_output
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.turns: list[Turn] = []
        self.summary = ""
        self._evicted = 0

    # -- budget ------------------------------------------------------------

    @property
    def max_summary_tokens(self) -> int:
        """The summary competes with the turns for the same budget, so it has to
        be bounded relative to it. Unbounded, it grew every eviction until the
        budget went *negative* and the next add() evicted the entire history."""
        return max(64, self.max_tokens // 8)

    @property
    def budget(self) -> int:
        used = self._count(self.system_prompt) + self._count(self.summary)
        return max(0, self.max_tokens - self.reserve - used)

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0

    def used_tokens(self) -> int:
        return sum(t.tokens for t in self.turns)

    # -- mutation ----------------------------------------------------------

    def add(self, role: str, content: str, **kwargs) -> Turn:
        turn = Turn(role=role, content=content, tokens=self._count(content), **kwargs)
        self.turns.append(turn)
        self._enforce_budget()
        return turn

    def add_tool_result(self, tool_name: str, result: str) -> Turn:
        return self.add("tool", result, tool_name=tool_name)

    def pin(self, index: int) -> None:
        self.turns[index].pinned = True

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self._evicted = 0

    # -- eviction ----------------------------------------------------------

    def _priority(self, turn: Turn, position: int, total: int) -> float:
        """Higher survives. Recency dominates, but not absolutely."""
        if turn.pinned:
            return float("inf")
        score = position / max(1, total - 1)          # 0 oldest, 1 newest
        if turn.role == "user":
            score += 0.35        # the ask matters more than the answer
        elif turn.role == "tool":
            score += 0.10        # results are often re-derivable
        if position >= total - 4:
            score += 1.0         # never evict the last few; it breaks coherence
        if turn.tokens > 4000:
            score -= 0.25        # big turns are expensive to keep
        return score

    def _enforce_budget(self) -> None:
        if self.used_tokens() <= self.budget:
            return

        before = len(self.turns)
        if self.strategy == "truncate":
            while self.turns and self.used_tokens() > self.budget:
                dropped = self.turns.pop(0)
                self._evicted += 1
                if dropped.pinned:
                    self.turns.insert(0, dropped)   # can't drop it; stop trying
                    break

        elif self.strategy == "summarize":
            dropped: list[Turn] = []
            while self.turns and self.used_tokens() > self.budget:
                if self.turns[0].pinned:
                    break
                dropped.append(self.turns.pop(0))
                self._evicted += 1
            if dropped:
                self._fold_into_summary(dropped)

        else:  # priority
            total = len(self.turns)
            ranked = sorted(
                range(total), key=lambda i: self._priority(self.turns[i], i, total)
            )
            victims: set[int] = set()
            running = self.used_tokens()
            for idx in ranked:
                if running <= self.budget:
                    break
                if self.turns[idx].pinned:
                    continue
                victims.add(idx)
                running -= self.turns[idx].tokens
            if victims:
                kept = [t for i, t in enumerate(self.turns) if i not in victims]
                self._fold_into_summary([self.turns[i] for i in sorted(victims)])
                self.turns = kept
                self._evicted += len(victims)

        if len(self.turns) < before:
            log.debug(
                "memory: evicted %d turn(s) via %s, %d/%d tokens",
                before - len(self.turns),
                self.strategy,
                self.used_tokens(),
                self.budget,
            )

    def _fold_into_summary(self, turns: list[Turn]) -> None:
        """Extractive placeholder.

        The real implementation calls the model with a summarization prompt. That
        costs a forward pass mid-turn, which is why it's behind the strategy flag
        and why this fallback exists at all — it is better than nothing and worse
        than the real thing.
        """
        fragments = []
        for t in turns:
            head = t.content.strip().replace("\n", " ")
            if len(head) > 160:
                head = head[:157] + "..."
            fragments.append(f"{t.role}: {head}")
        addition = " | ".join(fragments)
        self.summary = (self.summary + " | " + addition).strip(" |") if self.summary else addition
        self._trim_summary()

    def _trim_summary(self) -> None:
        """Keep the summary under its token bound, dropping the oldest fragments.

        Trimming on characters is not sufficient — the bound that matters is
        tokens, and the ratio between them varies enough that a character cap
        let the summary blow through its token budget.
        """
        cap = self.max_summary_tokens
        if self._count(self.summary) <= cap:
            return

        fragments = self.summary.split(" | ")
        while len(fragments) > 1 and self._count(" | ".join(fragments)) > cap:
            fragments.pop(0)
        self.summary = " | ".join(fragments)

        # A single fragment can still be over budget; cut it by characters.
        while self.summary and self._count(self.summary) > cap:
            self.summary = self.summary[: int(len(self.summary) * 0.8)]

    # -- rendering ---------------------------------------------------------

    def messages(self) -> list[dict]:
        out: list[dict] = []
        system = self.system_prompt
        if self.summary:
            system = (
                f"{system}\n\n[Earlier conversation, condensed]\n{self.summary}"
                if system
                else f"[Earlier conversation, condensed]\n{self.summary}"
            )
        if system:
            out.append({"role": "system", "content": system})
        out.extend(t.to_message() for t in self.turns)
        return out

    def stats(self) -> dict:
        return {
            "turns": len(self.turns),
            "tokens": self.used_tokens(),
            "budget": self.budget,
            "evicted": self._evicted,
            "summarized": bool(self.summary),
            "strategy": self.strategy,
        }
