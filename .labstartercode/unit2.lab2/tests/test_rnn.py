import pytest
import torch

from ft4.models.rnn import Rnn
from ft4.pipeline.fixed_batch_data_module import FixedBatchDataModule
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from test_helpers import (
    assert_causal,
    assert_eval_deterministic,
    ce_from_logits,
    evaluate_ce,
    load_ckpt_cpu,
    make_tokens,
    make_trainer,
    set_seed,
    stories_val_batch,
    xy_from_tokens,
)


def _small_rnn(V: int = 32, pad: int = 0, dim: int = 64) -> Rnn:
    return Rnn(vocab_size=V, padding_idx=pad, dim=dim, expansion_factor=2, depth=2, lr=3e-3)


def test_forward_backward_smoke():
    set_seed(1234)
    V, pad, B, L = 32, 0, 4, 12
    model = _small_rnn(V=V, pad=pad)

    tokens = make_tokens(B, L+1, V, avoid_id=pad, seed=1234)
    x, y = xy_from_tokens(tokens)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (B, L, V)
    assert torch.isfinite(logits).all()

    model.train()
    logits = model(x)
    loss = ce_from_logits(logits, y, ignore_index=pad)
    assert torch.isfinite(loss).item()
    loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum().item())
    assert total > 0.0


@pytest.mark.filterwarnings("ignore:.*worker.*")
@pytest.mark.filterwarnings("ignore:.*logger.*")
@pytest.mark.filterwarnings(r"ignore:.*isinstance\(treespec, LeafSpec\).*is deprecated.*") #type:ignore
def test_overfit_and_memorize_a_single_batch():
    set_seed(1234)
    # RNN is sequential and harder to overfit a single batch than the n-gram MLP, so
    # we use a smaller, shorter batch and run for more steps.
    V, pad, B, L = 32, 0, 8, 12
    model = _small_rnn(V=V, pad=pad)

    train_tokens = make_tokens(B, L+1, V, avoid_id=pad, seed=1234)
    val_tokens = train_tokens.clone()

    dm = FixedBatchDataModule(train_tokens, val_tokens)
    trainer = make_trainer(max_steps=500)
    trainer.fit(model, datamodule=dm)

    model.eval()
    train_ce = evaluate_ce(model, train_tokens, ignore_index=pad)
    val_ce = evaluate_ce(model, val_tokens, ignore_index=pad)
    assert train_ce < 2.0, f"train CE too high: {train_ce:.3f}"
    assert val_ce < 2.0, f"val CE too high: {val_ce:.3f}"


# -------- Architecture tests (no checkpoint required) --------

def test_eval_deterministic():
    set_seed(1234)
    V, pad, B, L = 32, 0, 2, 16
    model = _small_rnn(V=V, pad=pad)
    x = make_tokens(B, L, V, avoid_id=pad, seed=1234)
    assert_eval_deterministic(model, x)


def test_causal():
    set_seed(1234)
    V, pad, B, L = 32, 0, 2, 16
    model = _small_rnn(V=V, pad=pad)
    x = make_tokens(B, L, V, avoid_id=pad, seed=1234)
    assert_causal(model, x)


# -------- Checkpoint tests (require a trained model) --------

@pytest.mark.needs_checkpoint("rnn")
def test_ckpt_val_ce(require_checkpoint):
    ckpt = require_checkpoint()
    model = load_ckpt_cpu(ckpt, Rnn)
    model.eval()

    tokens = stories_val_batch(batch_size=8, seq_len=64, data_size="small")
    ce = evaluate_ce(model, tokens)
    MAX_ACCEPTABLE_CE = 2.8  # Between NeuralNgram and Transformer; tune to your training.
    assert ce < MAX_ACCEPTABLE_CE, f"val CE unexpectedly large: {ce:.3f} (max={MAX_ACCEPTABLE_CE})"
