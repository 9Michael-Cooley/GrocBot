import math

import pytest

from grokbot.inference.sampler import GenerationConfig, Sampler, softmax
from grokbot.inference.stream import StopChecker


def peaked(vocab: int, index: int, height: float = 10.0) -> list[float]:
    logits = [0.0] * vocab
    logits[index] = height
    return logits


def test_greedy_is_argmax():
    s = Sampler(seed=0)
    cfg = GenerationConfig(temperature=0.0)
    token, _ = s.sample(peaked(50, 37), [], cfg)
    assert token == 37


def test_greedy_is_deterministic():
    cfg = GenerationConfig(temperature=0.0)
    logits = [0.1 * i for i in range(64)]
    assert Sampler(0).sample(logits, [], cfg)[0] == Sampler(999).sample(logits, [], cfg)[0]


def test_same_seed_same_draw():
    logits = [0.05 * i for i in range(200)]
    cfg = GenerationConfig(temperature=1.0)
    a = [Sampler(42).sample(logits, [], cfg)[0] for _ in range(5)]
    b = [Sampler(42).sample(logits, [], cfg)[0] for _ in range(5)]
    assert a == b


def test_top_k_keeps_exactly_k():
    logits = [float(i) for i in range(100)]
    Sampler.top_k_filter(logits, 5)
    assert sum(1 for v in logits if v != -math.inf) == 5


def test_top_k_noop_when_k_exceeds_vocab():
    logits = [float(i) for i in range(10)]
    Sampler.top_k_filter(logits, 50)
    assert all(v != -math.inf for v in logits)


def test_top_p_always_keeps_at_least_one():
    logits = [1.0] * 100
    Sampler.top_p_filter(logits, 0.001)
    assert sum(1 for v in logits if v != -math.inf) >= 1


def test_top_p_keeps_the_nucleus():
    # One token holds ~all the mass; p=0.5 should keep only it.
    logits = peaked(50, 3, height=20.0)
    Sampler.top_p_filter(logits, 0.5)
    assert sum(1 for v in logits if v != -math.inf) == 1


def test_min_p_scales_with_confidence():
    flat = [1.0] * 20
    Sampler.min_p_filter(flat, 0.5)
    assert sum(1 for v in flat if v != -math.inf) == 20     # nothing to prune

    sharp = peaked(20, 0, height=8.0)
    Sampler.min_p_filter(sharp, 0.5)
    assert sum(1 for v in sharp if v != -math.inf) == 1


def test_repetition_penalty_pushes_seen_tokens_down():
    cfg = GenerationConfig(repetition_penalty=2.0)
    logits = [5.0, 5.0]
    Sampler.apply_penalties(logits, [0, 0, 0], cfg)
    assert logits[0] < logits[1]


def test_repetition_penalty_does_not_reward_negative_logits():
    """Multiplying throughout would make a negative logit *larger*."""
    cfg = GenerationConfig(repetition_penalty=2.0)
    logits = [-4.0, -4.0]
    Sampler.apply_penalties(logits, [0], cfg)
    assert logits[0] < logits[1]


def test_frequency_penalty_scales_with_count():
    cfg = GenerationConfig(frequency_penalty=1.0)
    logits = [10.0, 10.0]
    Sampler.apply_penalties(logits, [0, 0, 0, 1], cfg)
    assert logits[0] == pytest.approx(7.0)
    assert logits[1] == pytest.approx(9.0)


def test_penalty_window_limits_history():
    cfg = GenerationConfig(presence_penalty=5.0, penalty_window=2)
    logits = [10.0, 10.0]
    Sampler.apply_penalties(logits, [0, 1, 1], cfg)   # token 0 falls outside
    assert logits[0] == 10.0


def test_suppressed_tokens_are_never_drawn():
    s = Sampler(seed=1)
    cfg = GenerationConfig(temperature=1.0, suppress_tokens=[7])
    logits = peaked(20, 7, height=30.0)
    for _ in range(20):
        assert s.sample(logits, [], cfg)[0] != 7


def test_logit_bias_shifts_selection():
    s = Sampler(seed=0)
    cfg = GenerationConfig(temperature=0.0, logit_bias={11: 100.0})
    assert s.sample([0.0] * 40, [], cfg)[0] == 11


def test_softmax_survives_extreme_logits():
    probs = softmax([1000.0, 999.0, -1000.0])
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1.0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0.0)
    with pytest.raises(ValueError):
        GenerationConfig(max_tokens=0)


# -- stop strings -----------------------------------------------------------


def test_stop_checker_holds_partial_match():
    c = StopChecker(["END"])
    assert c.feed("hello E") == "hello "     # 'E' could still become 'END'
    assert c.feed("N") == ""
    assert c.feed("D") == ""
    assert c.stopped


def test_stop_checker_releases_false_partial():
    c = StopChecker(["END"])
    assert c.feed("hello E") == "hello "
    assert c.feed("agle") == "Eagle"          # not a stop after all
    assert not c.stopped


def test_stop_checker_truncates_at_stop():
    c = StopChecker(["STOP"])
    assert c.feed("keep thisSTOPdrop this") == "keep this"
    assert c.stopped
    assert c.feed("more") == ""


def test_stop_checker_picks_earliest_of_several():
    c = StopChecker(["BBB", "AAA"])
    assert c.feed("xxAAAyyBBB") == "xx"
    assert c.triggered == "AAA"


def test_stop_checker_ignores_empty_stops():
    c = StopChecker(["", "X"])
    assert c.feed("abc") == "abc"


def test_stop_checker_flush_releases_buffer():
    c = StopChecker(["END"])
    c.feed("tail EN")
    assert c.flush() == "EN"
