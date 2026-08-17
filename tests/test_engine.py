import pytest

from grokbot.errors import ContextLengthExceeded
from grokbot.inference.sampler import GenerationConfig
from grokbot.inference.stream import FinishReason


def test_generate_produces_text(engine):
    out = engine.generate("hello", GenerationConfig(max_tokens=24, seed=1))
    assert out.text
    assert out.completion_tokens > 0
    assert out.total_tokens == out.prompt_tokens + out.completion_tokens


def test_stream_matches_generate(engine):
    gen = GenerationConfig(max_tokens=24, seed=5)
    streamed = "".join(c.text for c in engine.stream("hello", gen))
    batched = engine.generate("hello", gen).text
    assert streamed == batched


def test_stream_ends_with_final_chunk(engine):
    chunks = list(engine.stream("hi", GenerationConfig(max_tokens=16, seed=2)))
    assert chunks[-1].is_final
    assert sum(1 for c in chunks if c.is_final) == 1


def test_seed_is_deterministic(engine):
    gen = GenerationConfig(max_tokens=32, temperature=0.9, seed=1234)
    assert engine.generate("same", gen).text == engine.generate("same", gen).text


def test_synthetic_backend_is_prompt_determined(engine):
    """The synthetic backend derives its stream from a hash of the prompt and
    emits a near-argmax spike, so the sampler seed does not change the output.
    That is a property of the stand-in backend, not of the sampler — sampler
    seeding is covered directly in test_sampler.py."""
    a = engine.generate("prompt", GenerationConfig(max_tokens=32, temperature=1.0, seed=1)).text
    b = engine.generate("prompt", GenerationConfig(max_tokens=32, temperature=1.0, seed=2)).text
    assert a == b

    c = engine.generate("a different prompt", GenerationConfig(max_tokens=32, seed=1)).text
    assert c != a


def test_max_tokens_respected(engine):
    out = engine.generate("hello", GenerationConfig(max_tokens=8, seed=3))
    assert out.completion_tokens <= 8


def test_stop_string_truncates(engine):
    full = engine.generate("hello", GenerationConfig(max_tokens=64, temperature=0.0)).text
    assert len(full) > 6, "need a non-trivial completion to slice a stop out of"

    marker = full[3:8]
    out = engine.generate(
        "hello", GenerationConfig(max_tokens=64, temperature=0.0, stop=[marker])
    )
    assert marker not in out.text
    assert out.finish_reason is FinishReason.STOP


def test_context_limit_enforced(engine):
    limit = engine.model_config.max_context()
    with pytest.raises(ContextLengthExceeded):
        engine.generate("x", GenerationConfig(max_tokens=limit + 10))


def test_empty_prompt_rejected(engine):
    with pytest.raises(ValueError, match="empty prompt"):
        engine.generate("", GenerationConfig(max_tokens=4))


def test_chat_applies_template(engine):
    out = engine.chat(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
        GenerationConfig(max_tokens=16, seed=9),
    )
    assert out.text
    assert "<|im_start|>" not in out.text     # control tokens must not leak out


def test_cache_is_released_between_requests(engine):
    for _ in range(6):
        engine.generate("release check", GenerationConfig(max_tokens=8, seed=1))
    snap = engine.cache.snapshot()
    assert snap["sequences"] == 0


def test_prefix_cache_engages_on_repeat(engine):
    prompt = "a shared prefix that is long enough to fill at least one block " * 4
    gen = GenerationConfig(max_tokens=4, seed=1)
    engine.generate(prompt, gen)
    engine.generate(prompt, gen)
    assert engine.cache.stats["cache_hits"] > 0


def test_scheduler_drains(engine):
    engine.generate("drain", GenerationConfig(max_tokens=8, seed=1))
    assert not engine.scheduler.has_work


def test_benchmark_step_runs(engine):
    stats = engine.benchmark_step(num_requests=4, prompt_tokens=32, output_tokens=8)
    assert stats["requests"] == 4
    assert stats["decode_tokens"] > 0
    assert stats["elapsed_s"] > 0


def test_stats_shape(engine):
    stats = engine.stats()
    assert stats["backend"] == "synthetic"
    assert "scheduler" in stats and "cache" in stats["scheduler"]
