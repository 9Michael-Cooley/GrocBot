import itertools

import pytest

from grokbot.errors import ContextLengthExceeded
from grokbot.inference.sampler import GenerationConfig
from grokbot.inference.scheduler import Request, Scheduler, SeqState
from grokbot.model.kv_cache import PagedKVCache


def make_scheduler(**kwargs):
    cache = PagedKVCache(num_blocks=kwargs.pop("num_blocks", 64), block_size=8)
    defaults = dict(max_batched_tokens=64, chunk_size=16, max_context=4096)
    defaults.update(kwargs)
    return Scheduler(cache, **defaults)


_prompt_seq = itertools.count()


def make_request(n_prompt=32, max_tokens=16, *, unique=True):
    """Distinct prompts by default.

    Identical prompts share every block through the prefix cache, so a test that
    reuses one can never create cache pressure no matter how many requests it
    queues. Pass unique=False when you *want* the sharing.
    """
    base = next(_prompt_seq) * 100_000 if unique else 0
    return Request(
        prompt_token_ids=list(range(base, base + n_prompt)),
        gen_config=GenerationConfig(max_tokens=max_tokens),
    )


def test_admits_waiting_request():
    s = make_scheduler()
    s.add_request(make_request())
    out = s.schedule()
    assert len(out.prefill) == 1
    assert s.running and not s.waiting


def test_chunked_prefill_splits_long_prompt():
    s = make_scheduler(chunk_size=16)
    req = make_request(n_prompt=64)
    s.add_request(req)

    out = s.schedule()
    assert out.chunk_sizes[req.request_id] == 16
    s.advance_prefill(req, 16)
    assert not req.prefill_done

    for _ in range(3):
        out = s.schedule()
        s.advance_prefill(req, out.chunk_sizes[req.request_id])
    assert req.prefill_done
    assert req.state is SeqState.RUNNING


def test_unchunked_prefill_takes_whole_prompt():
    s = make_scheduler(chunked_prefill=False, max_batched_tokens=128)
    req = make_request(n_prompt=100)
    s.add_request(req)
    out = s.schedule()
    assert out.chunk_sizes[req.request_id] == 100


def test_token_budget_caps_a_step():
    s = make_scheduler(max_batched_tokens=32, chunk_size=32)
    for _ in range(4):
        s.add_request(make_request(n_prompt=32))
    out = s.schedule()
    assert sum(out.chunk_sizes.values()) <= 32


def test_max_running_caps_concurrency():
    s = make_scheduler(max_running=2, max_batched_tokens=1024, chunk_size=8)
    for _ in range(5):
        s.add_request(make_request(n_prompt=8))
    s.schedule()
    assert len(s.running) <= 2


def test_context_length_is_enforced():
    s = make_scheduler(max_context=64)
    with pytest.raises(ContextLengthExceeded):
        s.add_request(make_request(n_prompt=60, max_tokens=32))


def test_preemption_folds_generated_tokens_into_the_prompt():
    """Recompute preemption discards KV but must not lose decoded tokens —
    they move into the prompt so the retry regenerates the same state."""
    s = make_scheduler(num_blocks=16)
    req = make_request(n_prompt=32)
    s.add_request(req)
    s.schedule()
    s.advance_prefill(req, 32)
    s.append_token(req, 999)

    victim = s._preempt()

    assert victim is req
    assert victim.state is SeqState.PREEMPTED
    assert victim.preemption_count == 1
    assert 999 in victim.prompt_token_ids
    assert victim.output_token_ids == []
    assert victim.prompt_len == 33
    assert s.waiting[0] is victim          # front of the queue, it's already old
    assert req not in s.running


def test_preemption_is_lifo():
    """GROK-4417: the most recently admitted sequence is the victim, which is
    what causes the thrashing above ~400 concurrent."""
    s = make_scheduler(num_blocks=64, max_batched_tokens=512, chunk_size=64)
    first, second = make_request(n_prompt=16), make_request(n_prompt=16)
    s.add_request(first)
    s.add_request(second)
    s.schedule()
    assert s._preempt() is second          # newest loses, not oldest


def test_preemption_returns_none_when_nothing_to_evict():
    assert make_scheduler()._preempt() is None


def test_thrashing_under_pressure_is_reproducible():
    """Documents the known failure mode rather than asserting it away."""
    s = make_scheduler(num_blocks=8, max_batched_tokens=256, chunk_size=64)
    for _ in range(8):
        s.add_request(make_request(n_prompt=32))
    for _ in range(6):
        s.schedule()
    assert s.stats["preemptions"] > 0
    assert any(r.preemption_count > 1 for r in s.waiting + s.running)


def test_swap_preemption_is_rejected():
    cache = PagedKVCache(num_blocks=8, block_size=8)
    with pytest.raises(NotImplementedError, match="unfinished"):
        Scheduler(cache, preemption="swap")


def test_abort_removes_and_frees():
    s = make_scheduler()
    req = make_request()
    s.add_request(req)
    s.schedule()
    assert s.abort(req.request_id)
    assert not s.running and not s.waiting
    assert not s.abort("does-not-exist")


def test_finish_moves_to_finished():
    s = make_scheduler()
    req = make_request()
    s.add_request(req)
    s.schedule()
    s.advance_prefill(req, req.prompt_len)
    s.finish(req)
    assert req.state is SeqState.FINISHED
    assert s.stats["completed"] == 1
    assert not s.has_work


def test_waiting_queue_bound():
    from grokbot.errors import OutOfCacheBlocks

    s = make_scheduler(max_waiting=2)
    s.add_request(make_request())
    s.add_request(make_request())
    with pytest.raises(OutOfCacheBlocks, match="waiting queue full"):
        s.add_request(make_request())


def test_ttft_recorded_on_first_token():
    s = make_scheduler()
    req = make_request()
    s.add_request(req)
    s.schedule()
    s.advance_prefill(req, req.prompt_len)
    assert req.ttft() is None
    s.append_token(req, 5)
    assert req.ttft() is not None and req.ttft() >= 0
