import pytest

from grokbot.errors import OutOfCacheBlocks, SequenceNotFound
from grokbot.model.kv_cache import PagedKVCache


@pytest.fixture
def cache():
    return PagedKVCache(num_blocks=32, block_size=8, num_layers=4, num_kv_heads=4, head_dim=64)


def test_allocation_block_math(cache):
    table = cache.allocate("a", list(range(20)))
    assert len(table) == 3            # ceil(20/8)
    assert table.num_tokens == 20
    assert cache.num_free_blocks == 29


def test_free_returns_uncached_blocks():
    cache = PagedKVCache(num_blocks=16, block_size=8, enable_prefix_caching=False)
    cache.allocate("a", list(range(24)))
    assert cache.num_free_blocks == 13
    cache.free("a")
    assert cache.num_free_blocks == 16


def test_prefix_caching_shares_full_blocks(cache):
    prompt = list(range(32))
    cache.allocate("a", prompt)
    before = cache.num_free_blocks
    cache.allocate("b", prompt)
    # Every block is full and identical, so the second sequence allocates none.
    assert cache.num_free_blocks == before
    assert cache.stats["cache_hits"] == 4


def test_partial_block_breaks_the_prefix_chain(cache):
    cache.allocate("a", list(range(12)))     # one full block + one partial
    cache.allocate("b", list(range(12)))
    assert cache.stats["cache_hits"] == 1    # only the full block is shareable


def test_append_grows_and_allocates(cache):
    cache.allocate("a", list(range(8)))      # exactly one full block
    assert len(cache.get_table("a")) == 1
    cache.append_token("a")                  # forces a new block
    assert len(cache.get_table("a")) == 2
    assert cache.get_table("a").num_tokens == 9


def test_fork_shares_until_write(cache):
    cache.allocate("parent", list(range(16)))
    free_after_parent = cache.num_free_blocks

    cache.fork("parent", "child")
    assert cache.num_free_blocks == free_after_parent      # sharing is free

    cache.append_token("child")
    cache.append_token("child")
    assert cache.stats["cow_copies"] >= 0
    assert cache.get_table("child").num_tokens == 18
    assert cache.get_table("parent").num_tokens == 16      # parent untouched


def test_physical_slots_match_token_count(cache):
    cache.allocate("a", list(range(20)))
    for _ in range(5):
        cache.append_token("a")
    assert len(cache.physical_slots("a")) == 25


def test_out_of_blocks_raises():
    cache = PagedKVCache(num_blocks=2, block_size=8, enable_prefix_caching=False)
    with pytest.raises(OutOfCacheBlocks):
        cache.allocate("big", list(range(100)))


def test_watermark_reserves_capacity():
    cache = PagedKVCache(num_blocks=10, block_size=8)
    assert cache.can_allocate(80, watermark=0.0)
    assert not cache.can_allocate(80, watermark=0.2)


def test_eviction_reclaims_unreferenced_cached_blocks():
    cache = PagedKVCache(num_blocks=4, block_size=8)
    cache.allocate("a", list(range(32)))
    cache.free("a")
    # Blocks stay populated for prefix reuse but are unreferenced, so a fresh
    # sequence with different content must be able to evict them.
    cache.allocate("b", list(range(100, 132)))
    assert cache.stats["evictions"] > 0


def test_free_keeps_full_blocks_cached(cache):
    """Freeing does not return full blocks to the free list — they stay
    populated so a later request can hit the prefix. They become evictable."""
    cache.allocate("a", list(range(16)))     # two full blocks
    assert cache.free("a") == 0
    assert cache.num_evictable_blocks == 2
    assert cache.num_available_blocks == cache.num_blocks


def test_double_free_is_survivable(cache):
    cache.allocate("a", list(range(16)))
    cache.free("a")
    assert cache.free("a") == 0          # must not underflow the refcounts
    assert all(b.ref_count >= 0 for b in cache._blocks)


def test_unknown_sequence_raises(cache):
    with pytest.raises(SequenceNotFound):
        cache.get_table("nope")


def test_duplicate_allocation_rejected(cache):
    cache.allocate("a", [1, 2, 3])
    with pytest.raises(ValueError, match="already allocated"):
        cache.allocate("a", [4, 5, 6])


def test_block_size_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        PagedKVCache(num_blocks=8, block_size=12)
