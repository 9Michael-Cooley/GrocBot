"""Input and output filters.

Pattern matching, nothing more. This layer catches the mechanical cases — a
credit card number in a completion, a credential in a prompt — and does not
attempt to judge intent. Anything requiring judgement is the policy model's job,
and the policy model is not in this tree (safety/policy.py calls out to it).

Two rules learned the hard way:

  Filters run on the *decoded* text, never on token ids. A pattern written
  against tokens misses anything the tokenizer split differently, and the
  tokenizer splits differently depending on the preceding byte.

  Output filtering runs on the accumulated buffer, not per chunk. A card number
  arriving as four chunks matches nothing on any individual chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging import get_logger

log = get_logger(__name__)


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class FilterHit:
    filter_name: str
    pattern_name: str
    action: Action
    span: tuple[int, int]
    excerpt: str = ""      # redacted before it reaches a log


@dataclass
class FilterResult:
    text: str
    hits: list[FilterHit] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(h.action is Action.BLOCK for h in self.hits)

    @property
    def modified(self) -> bool:
        return any(h.action is Action.REDACT for h in self.hits)


class Filter:
    name = "base"

    def apply(self, text: str) -> FilterResult:  # pragma: no cover - interface
        raise NotImplementedError


def _luhn(digits: str) -> bool:
    """Card-number checksum. Without it, any 16-digit string (order numbers,
    timestamps concatenated, hashes) trips the filter — that was ~90% of hits."""
    total, alt = 0, False
    for ch in reversed(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


class PIIFilter(Filter):
    name = "pii"

    PATTERNS: dict[str, tuple[re.Pattern, Action]] = {
        "email": (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), Action.REDACT),
        # Trailing digit is matched separately so the group can't consume the
        # separator *after* the number — it swallowed the following space and
        # redaction ran words together.
        "card": (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), Action.REDACT),
        "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), Action.REDACT),
        "phone": (re.compile(r"\b\+?\d{1,2}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"), Action.REDACT),
        "ipv4": (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), Action.ALLOW),
    }

    def apply(self, text: str) -> FilterResult:
        hits: list[FilterHit] = []
        out = text

        for pname, (pattern, action) in self.PATTERNS.items():
            if action is Action.ALLOW:
                continue
            for match in list(pattern.finditer(out)):
                raw = match.group()
                if pname == "card" and not _luhn(re.sub(r"[ -]", "", raw)):
                    continue
                hits.append(
                    FilterHit(self.name, pname, action, match.span(), excerpt=_mask(raw))
                )
            out = pattern.sub(lambda m: _mask(m.group()) if _keep(pname, m.group()) else m.group(), out)
        return FilterResult(out, hits)


def _keep(pname: str, raw: str) -> bool:
    if pname == "card":
        return _luhn(re.sub(r"[ -]", "", raw))
    return True


def _mask(raw: str) -> str:
    if "@" in raw:
        user, _, domain = raw.partition("@")
        return f"{user[0]}***@{domain}"
    digits = re.sub(r"\D", "", raw)
    return f"[redacted-{len(digits)}-digits]" if digits else "[redacted]"


class InjectionFilter(Filter):
    """Heuristics for prompt injection in retrieved/tool content.

    These are weak by construction. An attacker who knows the patterns writes
    around them in one try. The value is catching the untargeted case — a page
    that got scraped with a generic 'ignore previous instructions' block on it —
    and raising the cost slightly for everyone else. Do not treat a clean pass
    as evidence that content is safe.
    """

    name = "injection"

    PATTERNS = [
        ("override", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I)),
        ("role_reset", re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I)),
        ("system_spoof", re.compile(r"<\|im_start\|>\s*system", re.I)),
        ("exfil", re.compile(r"(print|reveal|repeat|output)\s+(your|the)\s+(system\s+)?prompt", re.I)),
        ("delimiter", re.compile(r"-{3,}\s*(end|stop)\s+of\s+(prompt|instructions?)", re.I)),
    ]

    def apply(self, text: str) -> FilterResult:
        hits = [
            FilterHit(self.name, pname, Action.BLOCK, m.span(), excerpt=m.group()[:60])
            for pname, pattern in self.PATTERNS
            for m in pattern.finditer(text)
        ]
        return FilterResult(text, hits)


class LeakageFilter(Filter):
    """Catches the model reciting its own system prompt or control tokens."""

    name = "leakage"

    PATTERNS = [
        ("control_token", re.compile(r"<\|(im_start|im_end|tool_call|think)\|>")),
        ("api_key", re.compile(r"\b(sk|xai|gsk)-[A-Za-z0-9]{16,}\b")),
        ("private_key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ]

    def apply(self, text: str) -> FilterResult:
        hits: list[FilterHit] = []
        out = text
        for pname, pattern in self.PATTERNS:
            for m in pattern.finditer(out):
                action = Action.BLOCK if pname != "control_token" else Action.REDACT
                hits.append(FilterHit(self.name, pname, action, m.span(), excerpt="[redacted]"))
            if pname == "control_token":
                out = pattern.sub("", out)
        return FilterResult(out, hits)


_FILTERS: dict[str, type[Filter]] = {
    "pii": PIIFilter,
    "injection": InjectionFilter,
    "leakage": LeakageFilter,
}


def build_filters(names: list[str]) -> list[Filter]:
    out = []
    for n in names or []:
        cls = _FILTERS.get(n)
        if cls is None:
            log.warning("unknown filter %r; skipping", n)
            continue
        out.append(cls())
    return out


def run_filters(text: str, filters: list[Filter]) -> FilterResult:
    hits: list[FilterHit] = []
    current = text
    for f in filters:
        result = f.apply(current)
        current = result.text
        hits.extend(result.hits)
        if result.blocked:
            break       # no point running the rest
    return FilterResult(current, hits)
