import pytest
import torch

from ft4.models.neural_ngram import NeuralNgram
from ft4.pipeline.fixed_batch_data_module import FixedBatchDataModule
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from test_helpers import *
from test_helpers import (
    assert_causal,
    assert_eval_deterministic,
    stories_val_batch,
)

@pytest.mark.lab("unit1.lab2")
def test_forward_backward_smoke():
    set_seed(1234)
    V, pad, B, L = 32, 0, 4, 12
    model = NeuralNgram(n=3, vocab_size=V, padding_idx=pad, dim=64, depth=1, expansion_factor=2, lr=3e-3)

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


@pytest.mark.lab("unit1.lab2")
@pytest.mark.filterwarnings("ignore:.*worker.*")
@pytest.mark.filterwarnings("ignore:.*logger.*")
@pytest.mark.filterwarnings("ignore:.*treespec, LeafSpec.*is deprecated.*") #type:ignore
def test_overfit_and_memorize_a_single_batch():
    set_seed(1234)
    V, pad, B, L = 32, 0, 16, 32
    model = NeuralNgram(n=3, vocab_size=V, padding_idx=pad, dim=64, depth=1, expansion_factor=2, lr=3e-3)

    train_tokens = make_tokens(B, L+1, V, avoid_id=pad, seed=1234)
    val_tokens   = train_tokens.clone()

    dm = FixedBatchDataModule(train_tokens, val_tokens)
    trainer = make_trainer(max_steps=200)
    trainer.fit(model, datamodule=dm)

    model.eval()
    train_ce = evaluate_ce(model, train_tokens, ignore_index=pad)
    val_ce   = evaluate_ce(model, val_tokens, ignore_index=pad)
    assert train_ce < 2.0, f"train CE too high: {train_ce:.3f}"
    assert val_ce   < 2.0, f"val CE too high: {val_ce:.3f}"


# -------- Architecture tests (no checkpoint required) --------

@pytest.mark.lab("unit1.lab2")
def test_eval_deterministic():
    set_seed(1234)
    V, pad, B, L = 32, 0, 2, 16
    model = NeuralNgram(n=3, vocab_size=V, padding_idx=pad, dim=64, depth=1, expansion_factor=2, lr=3e-3)
    x = make_tokens(B, L, V, avoid_id=pad, seed=1234)
    assert_eval_deterministic(model, x)


@pytest.mark.lab("unit1.lab2")
def test_causal():
    set_seed(1234)
    V, pad, B, L = 32, 0, 2, 16
    model = NeuralNgram(n=3, vocab_size=V, padding_idx=pad, dim=64, depth=1, expansion_factor=2, lr=3e-3)
    x = make_tokens(B, L, V, avoid_id=pad, seed=1234)
    assert_causal(model, x)


# -------- Checkpoint tests (require a trained model) --------

@pytest.mark.lab("unit1.lab2")
@pytest.mark.needs_checkpoint("neural_ngram")
def test_ckpt_val_ce(require_checkpoint):
    ckpt = require_checkpoint()
    model = load_ckpt_cpu(ckpt, NeuralNgram)
    model.eval()

    tokens = stories_val_batch(batch_size=8, seq_len=64, data_size="small")
    ce = evaluate_ce(model, tokens)
    MAX_ACCEPTABLE_CE = 3.0  # NeuralNgram (n=3) is weak; adjust to your hardware/training time.
    assert ce < MAX_ACCEPTABLE_CE, f"val CE unexpectedly large: {ce:.3f} (max={MAX_ACCEPTABLE_CE})"
