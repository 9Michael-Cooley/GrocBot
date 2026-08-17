"""Deterministic RNG.

Reproducibility matters more here than speed. Python's `random` is seeded per
process and gets perturbed by anything else that touches it, so we carry our own
splittable generator: same seed + same sequence id => same stream, regardless of
what else is running or in what order.

xoshiro256** over a SplitMix64-expanded seed. Not cryptographic. Do not use it
for anything that needs to be unguessable.
"""

from __future__ import annotations

_MASK = (1 << 64) - 1


def _splitmix64(state: int):
    def nxt() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    return nxt


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK


class Rng:
    """xoshiro256**. Cheap to split, stable across runs and platforms."""

    __slots__ = ("s",)

    def __init__(self, seed: int = 0):
        seeder = _splitmix64(seed & _MASK)
        self.s = [seeder() for _ in range(4)]

    def next_u64(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & _MASK, 7) * 9) & _MASK
        t = (s[1] << 17) & _MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def random(self) -> float:
        """Uniform in [0, 1). 53 bits of mantissa, same convention as CPython."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive on both ends. Rejection-sampled so it stays unbiased."""
        span = hi - lo + 1
        if span <= 0:
            raise ValueError(f"empty range [{lo}, {hi}]")
        limit = _MASK - (_MASK % span)
        while True:
            v = self.next_u64()
            if v <= limit:
                return lo + (v % span)

    def choice(self, seq):
        if not seq:
            raise IndexError("choice from empty sequence")
        return seq[self.randint(0, len(seq) - 1)]

    def shuffle(self, seq: list) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Marsaglia polar. Discards the second variate — we rarely want pairs."""
        import math

        while True:
            u = 2.0 * self.random() - 1.0
            v = 2.0 * self.random() - 1.0
            q = u * u + v * v
            if 0.0 < q < 1.0:
                return mu + sigma * u * math.sqrt(-2.0 * math.log(q) / q)

    def split(self, stream_id: int) -> Rng:
        """Independent child stream. Parent is untouched."""
        mixed = _splitmix64((self.s[0] ^ (stream_id * 0x9E3779B97F4A7C15)) & _MASK)
        child = Rng.__new__(Rng)
        child.s = [mixed() for _ in range(4)]
        return child

    def fork(self) -> Rng:
        return self.split(self.next_u64())


def seed_from_text(text: str) -> int:
    """FNV-1a. Used to make synthetic generation depend on the prompt."""
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & _MASK
    return h
