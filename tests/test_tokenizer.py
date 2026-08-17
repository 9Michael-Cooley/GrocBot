import pytest

from grokbot.tokenizer import Tokenizer, render_chat, strip_thinking
from grokbot.tokenizer.bpe import _SPLIT_RE
from grokbot.tokenizer.special import IM_END, THINK_END, THINK_START

ROUNDTRIP_CASES = [
    "",
    "a",
    "Hello, world!",
    "why is the sky blue?",
    "emoji 🚀 and ünïcode ﷽",
    "tabs\tand\nnewlines   ",
    "1234567890 !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "under_scores_everywhere",
    "   leading and trailing   ",
    "日本語のテキスト",
    "a" * 500,
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_pretokenizer_is_total(text):
    """Every character must match some alternative, or encoding drops it."""
    assert "".join(_SPLIT_RE.findall(text)) == text


def test_special_tokens_roundtrip(tokenizer):
    text = f"before{IM_END}after"
    ids = tokenizer.encode(text)
    assert tokenizer.specials[IM_END] in ids
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special=True) == "beforeafter"


def test_incremental_decode_matches_full(tokenizer):
    """Streaming must reassemble to exactly the batch decoding."""
    text = "streaming 🚀 with ünïcode and 日本語"
    ids = tokenizer.encode(text)

    assembled, prev = "", 0
    for i in range(1, len(ids) + 1):
        assembled += tokenizer.decode_incremental(ids[:i], prev)
        prev = i
    assert assembled == text


def test_incremental_never_emits_replacement_char(tokenizer):
    ids = tokenizer.encode("🚀🎉🔥")
    prev = 0
    for i in range(1, len(ids) + 1):
        assert "�" not in tokenizer.decode_incremental(ids[:i], prev)
        prev = i


def test_unknown_token_id_raises(tokenizer):
    from grokbot.errors import TokenizerError

    with pytest.raises(TokenizerError, match="not in vocab"):
        tokenizer.decode([tokenizer.vocab_size + 999])


def test_synthetic_is_deterministic():
    a = Tokenizer.synthetic(vocab_size=2048, seed=7)
    b = Tokenizer.synthetic(vocab_size=2048, seed=7)
    assert a.encode("determinism matters") == b.encode("determinism matters")


def test_render_chat_roles():
    out = render_chat([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    assert "<|im_start|>system" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_render_chat_rejects_bad_role():
    with pytest.raises(ValueError, match="unknown role"):
        render_chat([{"role": "wizard", "content": "x"}])


def test_strip_thinking():
    visible, thought = strip_thinking(f"a{THINK_START}hidden{THINK_END}b")
    assert visible == "ab"
    assert thought == "hidden"


def test_strip_thinking_unterminated_is_all_reasoning():
    visible, thought = strip_thinking(f"a{THINK_START}cut off here")
    assert visible == "a"
    assert thought == "cut off here"
