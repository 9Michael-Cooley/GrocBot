"""Paged KV cache.

Attention state lives in fixed-size blocks and each sequence holds a block
table, so a sequence's memory is non-contiguous and we stop paying for the
padding a contiguous layout forces. Blocks are refcounted, which makes
copy-on-write forking (beam search, n-best, speculative drafts) and prefix
sharing across requests fall out for free.

Fragmentation is bounded by block_size - 1 tokens per sequence, worst case.

Known issues:
  - eviction scan is O(n) over cached blocks. Fine at current scale. GROK-4417
    tracks the replacement.
  - refcount underflow was possible when a sequence was freed twice; there's an
    assert on the path now but the double-free root cause was never found.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..errors import OutOfCacheBlocks, SequenceNotFound
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    num_tokens: int = 0          # slots filled; < block_size means partial
    content_hash: str | None = None   # set only when full and cacheable
    last_used: int = 0

    @property
    def is_full(self) -> bool:
        return self.content_hash is not None


@dataclass
class BlockTable:
    seq_id: str
    blocks: list[Block] = field(default_factory=list)
    num_tokens: int = 0
    num_cached: int = 0          # prefix tokens served from the shared cache

    def __len__(self) -> int:
        return len(self.blocks)


class PagedKVCache:
    """Block allocator. Holds no tensors — the backend owns the actual storage."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        *,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: str = "bf16",
        enable_prefix_caching: bool = True,
    ):
        if block_size & (block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {block_size}")

        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.enable_prefix_caching = enable_prefix_caching

        self._blocks = [Block(i) for i in range(num_blocks)]
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._tables: dict[str, BlockTable] = {}
        self._hash_index: dict[str, int] = {}   # content hash -> block id
        self._clock = 0

        self.stats = {
            "allocated": 0,
            "freed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "cow_copies": 0,
        }

    # -- capacity ----------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_evictable_blocks(self) -> int:
        """Unreferenced blocks still holding cached content.

        These are not on the free list — they're kept populated so a later
        request can hit the prefix — but they can be reclaimed on demand.
        """
        return sum(1 for b in self._blocks if b.ref_count == 0 and b.content_hash is not None)

    @property
    def num_available_blocks(self) -> int:
        return self.num_free_blocks + self.num_evictable_blocks

    @property
    def utilization(self) -> float:
        """Fraction of blocks that are pinned. Cached-but-unreferenced blocks
        are not "in use" — counting them made the gauge read 100% permanently
        once prefix caching had warmed up."""
        if not self.num_blocks:
            return 1.0
        pinned = sum(1 for b in self._blocks if b.ref_count > 0)
        return pinned / self.num_blocks

    def bytes_per_block(self) -> int:
        width = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}.get(self.dtype, 2)
        # 2 = one K plane and one V plane
        return 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim * width

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int, *, watermark: float = 0.0) -> bool:
        # Evictable blocks count as available. Checking only the free list made
        # allocation fail while reclaimable capacity sat idle, which the
        # scheduler then answered by preempting a live sequence for no reason.
        need = self.blocks_needed(num_tokens)
        reserve = int(self.num_blocks * watermark)
        return self.num_available_blocks - need >= reserve

    # -- hashing -----------------------------------------------------------

    def _hash(self, token_ids: list[int], prefix_hash: str | None) -> str:
        h = hashlib.blake2b(digest_size=16)
        if prefix_hash:
            h.update(prefix_hash.encode("ascii"))
        h.update(b",".join(str(t).encode("ascii") for t in token_ids))
        return h.hexdigest()

    # -- allocation --------------------------------------------------------

    def _take_free_block(self) -> Block:
        if self._free:
            return self._blocks[self._free.pop()]
        victim = self._evict()
        if victim is None:
            raise OutOfCacheBlocks(
                f"no free blocks ({self.num_blocks} total, "
                f"{sum(1 for b in self._blocks if b.ref_count) } pinned)"
            )
        return victim

    def _evict(self) -> Block | None:
        """LRU over unreferenced cached blocks. O(n); see module docstring."""
        best: Block | None = None
        for blk in self._blocks:
            if blk.ref_count == 0 and blk.content_hash is not None:
                if best is None or blk.last_used < best.last_used:
                    best = blk
        if best is None:
            return None
        self._hash_index.pop(best.content_hash, None)  # type: ignore[arg-type]
        best.content_hash = None
        best.num_tokens = 0
        self.stats["evictions"] += 1
        return best

    def allocate(self, seq_id: str, token_ids: list[int], *, watermark: float = 0.0) -> BlockTable:
        """Reserve blocks for a prompt, reusing any shared prefix."""
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id!r} already allocated")

        if not self.can_allocate(len(token_ids), watermark=watermark):
            # Caller (scheduler) decides whether to preempt or requeue.
            raise OutOfCacheBlocks(
                f"need {self.blocks_needed(len(token_ids))} blocks, {self.num_free_blocks} free"
            )

        table = BlockTable(seq_id=seq_id)
        prefix_hash: str | None = None

        for start in range(0, len(token_ids), self.block_size):
            chunk = token_ids[start : start + self.block_size]
            full = len(chunk) == self.block_size

            if full and self.enable_prefix_caching:
                chunk_hash = self._hash(chunk, prefix_hash)
                hit = self._hash_index.get(chunk_hash)
                if hit is not None:
                    blk = self._blocks[hit]
                    blk.ref_count += 1
                    blk.last_used = self._clock
                    table.blocks.append(blk)
                    table.num_cached += len(chunk)
                    self.stats["cache_hits"] += 1
                    prefix_hash = chunk_hash
                    continue
                self.stats["cache_misses"] += 1

            blk = self._take_free_block()
            blk.ref_count = 1
            blk.num_tokens = len(chunk)
            blk.last_used = self._clock
            if full and self.enable_prefix_caching:
                blk.content_hash = self._hash(chunk, prefix_hash)
                self._hash_index[blk.content_hash] = blk.block_id
                prefix_hash = blk.content_hash
            else:
                blk.content_hash = None
                prefix_hash = None  # partial block breaks the chain
            table.blocks.append(blk)
            self.stats["allocated"] += 1

        table.num_tokens = len(token_ids)
        self._tables[seq_id] = table
        self._clock += 1
        return table

    def append_token(self, seq_id: str) -> None:
        """Grow a sequence by one decoded token, adding a block if needed."""
        table = self._tables.get(seq_id)
        if table is None:
            raise SequenceNotFound(f"no block table for {seq_id!r}")

        offset = table.num_tokens % self.block_size
        if offset == 0:
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.num_tokens = 0
            blk.content_hash = None
            table.blocks.append(blk)
            self.stats["allocated"] += 1

        last = table.blocks[-1]
        if last.ref_count > 1:
            # Shared block being written to — copy first.
            last = self._copy_on_write(table, len(table.blocks) - 1)
        last.num_tokens += 1
        last.last_used = self._clock
        table.num_tokens += 1
        self._clock += 1

    def _copy_on_write(self, table: BlockTable, index: int) -> Block:
        old = table.blocks[index]
        new = self._take_free_block()
        new.ref_count = 1
        new.num_tokens = old.num_tokens
        new.content_hash = None      # diverged; no longer shareable
        old.ref_count -= 1
        table.blocks[index] = new
        self.stats["cow_copies"] += 1
        self.stats["allocated"] += 1
        return new

    def fork(self, parent_id: str, child_id: str) -> BlockTable:
        """Share a parent's blocks with a child. Writes trigger CoW."""
        parent = self._tables.get(parent_id)
        if parent is None:
            raise SequenceNotFound(f"no block table for {parent_id!r}")
        for blk in parent.blocks:
            blk.ref_count += 1
        child = BlockTable(
            seq_id=child_id,
            blocks=list(parent.blocks),
            num_tokens=parent.num_tokens,
            num_cached=parent.num_cached,
        )
        self._tables[child_id] = child
        return child

    def free(self, seq_id: str) -> int:
        """Release a sequence. Returns blocks actually returned to the pool."""
        table = self._tables.pop(seq_id, None)
        if table is None:
            log.debug("free() on unknown sequence %s (already freed?)", seq_id)
            return 0

        released = 0
        for blk in table.blocks:
            assert blk.ref_count > 0, f"refcount underflow on block {blk.block_id}"
            blk.ref_count -= 1
            if blk.ref_count == 0:
                if blk.content_hash is None:
                    blk.num_tokens = 0
                    self._free.append(blk.block_id)
                    released += 1
                # else: keep it populated so a later request can hit the prefix
        self.stats["freed"] += released
        return released

    def get_table(self, seq_id: str) -> BlockTable:
        table = self._tables.get(seq_id)
        if table is None:
            raise SequenceNotFound(f"no block table for {seq_id!r}")
        return table

    def physical_slots(self, seq_id: str) -> list[int]:
        """Flatten a block table to per-token physical slot indices."""
        table = self.get_table(seq_id)
        slots: list[int] = []
        for i, blk in enumerate(table.blocks):
            count = blk.num_tokens if i == len(table.blocks) - 1 else self.block_size
            base = blk.block_id * self.block_size
            slots.extend(range(base, base + count))
        return slots

    def reset(self) -> None:
        for blk in self._blocks:
            blk.ref_count = 0
            blk.num_tokens = 0
            blk.content_hash = None
        self._free = list(reversed(range(self.num_blocks)))
        self._tables.clear()
        self._hash_index.clear()

    def snapshot(self) -> dict:
        return {
            **self.stats,
            "num_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "utilization": round(self.utilization, 4),
            "sequences": len(self._tables),
            "cached_blocks": len(self._hash_index),
        }
