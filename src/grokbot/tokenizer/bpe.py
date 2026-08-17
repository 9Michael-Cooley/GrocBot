"""Byte-level BPE.

Same construction as GPT-2/Llama: every input byte maps to a base token, so the
vocabulary is closed over arbitrary bytes and encode/decode round-trips exactly
for any input, including invalid UTF-8. Merges are applied greedily by rank.

The production vocab ships inside the checkpoint (`tokenizer.json`) and is not
vendored here. Without it, `Tokenizer.synthetic()` builds a deterministic vocab
from a seeded fragment table. Token *ids* from the synthetic vocab are
meaningless — they will not match a real checkpoint. Round-tripping still works,
which is all the serving tests need.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ..errors import TokenizerError
from ..utils.rng import Rng

# GPT-2's pretokenizer split. Keeps contractions and leading spaces attached.
#
# Python's `re` has no \p{L}/\p{N}, so the Unicode categories are spelled out:
# [^\W\d_] is "word character that is neither digit nor underscore" == letter.
# The trailing `_+|.` alternatives make the pattern total — every character
# matches something. Without them, underscores and unpaired surrogates fall
# through findall() and silently vanish from the encoding. (They did. Twice.)
_SPLIT_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"| ?_+"
    r"|\s+(?!\S)"
    r"|\s+"
    r"|.",
    re.UNICODE | re.DOTALL,
)


@lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    """Reversible byte -> printable-codepoint map. Avoids control chars in vocab."""
    printable = list(range(ord("!"), ord("~") + 1))
    printable += list(range(ord("\xa1"), ord("\xac") + 1))
    printable += list(range(ord("\xae"), ord("\xff") + 1))
    mapped = printable[:]
    shift = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + shift)
            shift += 1
    return dict(zip(printable, (chr(c) for c in mapped)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class Tokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], specials: dict[str, int] | None = None):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.specials = specials or {}
        self.inv_specials = {v: k for k, v in self.specials.items()}

        self.byte_encoder = _byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache: dict[str, list[str]] = {}

        if self.specials:
            escaped = "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True))
            self._special_re: re.Pattern | None = re.compile(f"({escaped})")
        else:
            self._special_re = None

    # -- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.specials)

    def __len__(self) -> int:
        return self.vocab_size

    # -- core --------------------------------------------------------------

    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if len(word) < 2:
            self._cache[token] = list(word)
            return list(word)

        while True:
            candidates = _pairs(word)
            if not candidates:
                break
            best = min(candidates, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break

            first, second = best
            out: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    out.append(first + second)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
            if len(word) == 1:
                break

        result = list(word)
        self._cache[token] = result
        return result

    def encode(self, text: str, *, allowed_special: bool = True) -> list[int]:
        if not text:
            return []

        if self._special_re is not None and allowed_special:
            ids: list[int] = []
            for part in self._special_re.split(text):
                if not part:
                    continue
                if part in self.specials:
                    ids.append(self.specials[part])
                else:
                    ids.extend(self._encode_ordinary(part))
            return ids
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(mapped):
                tid = self.vocab.get(piece)
                if tid is None:
                    # Should be unreachable: every single byte is in the vocab.
                    for ch in piece:
                        ids.append(self.vocab[ch])
                else:
                    ids.append(tid)
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = False) -> str:
        buf = bytearray()
        out: list[str] = []
        for tid in ids:
            if tid in self.inv_specials:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special:
                    out.append(self.inv_specials[tid])
                continue
            piece = self.inv_vocab.get(tid)
            if piece is None:
                raise TokenizerError(f"token id {tid} not in vocab (size {self.vocab_size})")
            buf.extend(self.byte_decoder[c] for c in piece)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def decode_incremental(self, ids: list[int], prev_len: int) -> str:
        """Decode only what's new, without splitting a multi-byte character.

        Streaming needs this: decoding token-by-token emits replacement chars in
        the middle of a UTF-8 sequence or an emoji. We decode a suffix window and
        diff against the previous rendering.

        The trailing-U+FFFD strip on *both* sides is the important part. A
        partially-decoded character renders as one replacement char, so if only
        the `after` side were stripped the two strings would disagree on length
        at that position and the slice would eat real characters on the token
        that completes it. Symmetric stripping keeps the prefix aligned:

            "a" + [emoji lead byte]  ->  before "a�" -> "a",  after "a" -> ""
            "a" + [emoji complete]   ->  before "a�" -> "a",  after "a\U0001f680" -> emoji

        Caveat: a literal U+FFFD at the end of the model's own output is
        swallowed. Checkpoints don't emit one and it isn't worth the bookkeeping.

        An 8-token lookback is always enough — a UTF-8 character is at most 4
        bytes and every byte is at least one token.
        """
        window = max(0, prev_len - 8)
        before = self.decode(ids[window:prev_len], skip_special=True).rstrip("�")
        after = self.decode(ids[window:], skip_special=True).rstrip("�")
        if len(after) <= len(before):
            return ""  # mid-character, wait for the next token
        return after[len(before) :]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> Tokenizer:
        p = Path(path)
        if not p.exists():
            raise TokenizerError(
                f"tokenizer not found at {p}. The vocab ships with the checkpoint and is "
                f"not vendored in this repo; set weights.tokenizer or use Tokenizer.synthetic()."
            )
        blob = json.loads(p.read_text(encoding="utf-8"))
        try:
            vocab = blob["model"]["vocab"]
            merges = [tuple(m.split(" ", 1)) for m in blob["model"]["merges"]]
            specials = {t["content"]: t["id"] for t in blob.get("added_tokens", [])}
        except (KeyError, TypeError) as exc:
            raise TokenizerError(f"malformed tokenizer.json at {p}: {exc}") from exc
        return cls(vocab, merges, specials)

    @classmethod
    def synthetic(cls, vocab_size: int = 32768, seed: int = 0) -> Tokenizer:
        """Deterministic stand-in vocab. Ids do not match any real checkpoint."""
        from .special import SPECIAL_TOKENS

        enc = _byte_encoder()
        vocab: dict[str, int] = {enc[b]: b for b in range(256)}
        merges: list[tuple[str, str]] = []

        rng = Rng(seed)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        fragments = [enc[ord(c)] for c in alphabet + " "]

        # Grow merges by repeatedly joining existing pieces. Deterministic, and
        # produces a plausible length distribution without needing a corpus.
        next_id = 256
        pool = list(fragments)
        while next_id < vocab_size - len(SPECIAL_TOKENS):
            a = rng.choice(pool)
            b = rng.choice(pool if rng.random() < 0.7 else fragments)
            joined = a + b
            if joined in vocab or len(joined) > 16:
                continue
            vocab[joined] = next_id
            merges.append((a, b))
            if len(joined) <= 8:
                pool.append(joined)
            next_id += 1

        specials = {tok: next_id + i for i, tok in enumerate(SPECIAL_TOKENS)}
        return cls(vocab, merges, specials)
