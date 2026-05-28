"""Unit tests for LangGen.

Strategy: we control the input logits to control how much randomness each
test has to tolerate. Forbidden-token and top-k/top-p assertions are
statements about the *support* of the post-filter distribution; we craft
logits so that support is a known small set (or singleton), and assert
that draws land inside it. Temperature is parametrized across the
filtering tests because it doesn't change support — every assertion that
holds at T=1 holds at any T>0 with the same logits — so we get free
coverage of the temperature branch on every filter test.
"""
import pytest
import torch

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from ft4.lang_gen import LangGen


VOCAB_SIZE = Ft4Tokenizer.last_tid + 1
PAD = Ft4Tokenizer.RES_PAD
BOS = Ft4Tokenizer.RES_START
EOS = Ft4Tokenizer.RES_STOP

# Safe token IDs — picked to sit well clear of any special-token reserve.
SAFE_A, SAFE_B, SAFE_C, SAFE_D = 100, 101, 102, 103

# Temperatures swept across the filtering tests. 1.0 hits the no-op branch
# (sample() skips scaling when temp == 1.0); 0.5 and 1.5 both hit the
# active branch. Temperature doesn't change distribution support, so
# legal-set assertions are invariant under T at these values.
# Note: T=0 (greedy) isn't included — it triggers div-by-zero in the
# current implementation. See test_temperature_zero_is_greedy below.
TEMPS = [0.5, 1.0, 1.5]


class StubModel:
    """Minimal stand-in for a real model.

    Returns peaked logits with mass concentrated on `peak_token`, regardless
    of input. Shape (B, L, VOCAB_SIZE), broadcasting over whatever sequence
    length is passed in.
    """
    device = torch.device('cpu')

    def __init__(self, peak_token: int):
        self.peak_token = peak_token

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        logits = torch.full((B, L, VOCAB_SIZE), float('-inf'))
        logits[..., self.peak_token] = 100.0
        return logits


def make_logits_3d(values: dict[int, float]) -> torch.Tensor:
    """Construct a (1, 1, VOCAB_SIZE) logits tensor matching the shape
    sample() receives from the model.

    Every position is -inf by default; positions listed in `values` get
    the given finite logit. Using -inf for the negative space means those
    tokens have probability zero after softmax — the support of the
    distribution is exactly the keys of `values`.
    """
    logits = torch.full((1, 1, VOCAB_SIZE), float('-inf'))
    for tok_id, val in values.items():
        logits[..., tok_id] = val
    return logits


def make_gen(**kwargs) -> LangGen:
    """Construct a LangGen with a dummy model. The model is consulted
    only for `.device`; tests that exercise sample() directly never
    invoke it."""
    return LangGen(model=StubModel(peak_token=SAFE_A), tokenizer=Ft4Tokenizer, **kwargs)


# ----------------------------------------------------------------------
# Filtering: sample()
# ----------------------------------------------------------------------

@pytest.mark.parametrize("forbidden", [PAD, BOS])
@pytest.mark.parametrize("temperature", TEMPS)
def test_forbidden_tokens_never_sampled(forbidden, temperature):
    """The forbidden token holds the highest raw logit, but sample()
    must mask it before sampling. Only SAFE_A remains in the support,
    so the result must be SAFE_A — one draw suffices.
    """
    gen = make_gen(temperature=temperature)
    logits = make_logits_3d({forbidden: 100.0, SAFE_A: 0.0})
    assert gen.sample(logits).item() == SAFE_A

# Since we have no plans to implement top-k or top-p,
# these tests are commented out, but kept for reference
# @pytest.mark.skip(reason="top-k not implemented")
# @pytest.mark.parametrize("temperature", TEMPS)
# def test_top_k_restricts_support(temperature):
#     """Four legal tokens in strict descending logit order; top_k=2 must
#     confine sampling to the top two by logit.
#     """
#     gen = make_gen(top_k=2, temperature=temperature)
#     logits = make_logits_3d({SAFE_A: 5.0, SAFE_B: 4.0, SAFE_C: 3.0, SAFE_D: 2.0})
#     expected = {SAFE_A, SAFE_B}
#     for _ in range(10):
#         assert gen.sample(logits).item() in expected
# @pytest.mark.skip(reason="top-p not implemented")
# @pytest.mark.parametrize("temperature", TEMPS)
# def test_top_p_restricts_support(temperature):
#     """Regression test for the device-side-assert bug.
#     Sharply peaked distribution: SAFE_A holds essentially all the
#     probability mass. top_p=0.9 must keep only SAFE_A; the safety shift
#     in the sorted mask guarantees the top-1 stays even when its
#     probability alone already exceeds top_p. Input is 3D (1, 1, V) —
#     the exact shape sample() receives from the model, and the shape
#     that triggered the original bug.
#     """
#     gen = make_gen(top_p=0.9, temperature=temperature)
#     logits = make_logits_3d({SAFE_A: 100.0, SAFE_B: 0.0})
#     for _ in range(10):
#         assert gen.sample(logits).item() == SAFE_A


def test_temperature_smoke():
    """Temperature scaling is one line of arithmetic; the only failure
    mode worth catching is the branch breaking the pipeline."""
    gen = make_gen(temperature=0.5)
    logits = make_logits_3d({SAFE_A: 5.0, SAFE_B: 4.0})
    result = gen.sample(logits).item()
    assert 0 <= result < VOCAB_SIZE


def test_temperature_zero_is_greedy():
    """T=0 should bypass sampling and return argmax. Current code
    divides by T and crashes."""
    gen = make_gen(temperature=0.0)
    logits = make_logits_3d({SAFE_A: 5.0, SAFE_B: 4.0, SAFE_C: 3.0})
    for _ in range(5):
        assert gen.sample(logits).item() == SAFE_A

def test_min_p_filters_below_threshold():
    """Tokens with probability below `min_p * p_max` must be masked.
    Logits 5.0, 3.7, 3.0 produce softmax probs of roughly (0.7, 0.2, 0.1);
    with min_p=0.3 the threshold sits at 0.21, leaving only SAFE_A in
    the support.
    """
    gen = make_gen(min_p=0.3)
    logits = make_logits_3d({SAFE_A: 5.0, SAFE_B: 3.7, SAFE_C: 3.0})
    for _ in range(10):
        assert gen.sample(logits).item() == SAFE_A

def test_min_p_smoke():
    """min-p branch fires without breaking the pipeline."""
    gen = make_gen(min_p=0.1)
    logits = make_logits_3d({SAFE_A: 5.0, SAFE_B: 4.0})
    result = gen.sample(logits).item()
    assert 0 <= result < VOCAB_SIZE


# ----------------------------------------------------------------------
# Iteration: __next__()
# ----------------------------------------------------------------------

def test_bos_prepended_to_tokens():
    """initial_txt is tokenized and placed after a leading BOS."""
    gen = make_gen(initial_txt="hello")
    assert gen.tokens[0].item() == BOS
    assert gen.tokens[1:].tolist() == Ft4Tokenizer.tokenize("hello")


def test_iterator_stops_on_eos():
    """A model that emits only EOS must terminate iteration immediately."""
    gen = LangGen(model=StubModel(peak_token=EOS), tokenizer=Ft4Tokenizer)
    with pytest.raises(StopIteration):
        next(gen)


def test_iterator_stops_at_max_to_generate():
    """With max_to_generate=3, the iterator yields exactly 3 items
    and then halts."""
    gen = LangGen(
        model=StubModel(peak_token=SAFE_A),
        tokenizer=Ft4Tokenizer,
        max_to_generate=3,
    )
    items = list(gen)
    assert len(items) == 3
