import pytest
import math

import torch

from hardcoregenai.models.neural_ngram import NeuralNgram
from hardcoregenai.pipeline.fixed_batch_data_module import FixedBatchDataModule
from hardcoregenai.pipeline.bpe_tokenizer import HcgaiTokenizer
from hardcoregenai.util.test_helpers import *

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

@pytest.mark.lab("unit1.lab2")
@pytest.mark.needs_checkpoint
def test_ckpt_val_ce():
    ckpt = find_checkpoint()
    if ckpt is None:
        pytest.skip("No checkpoint found. To run this test, you must first train the model: use app/tell_stories_with_neural_ngram.py")

    model = load_ckpt_cpu(ckpt, NeuralNgram)
    model.eval()

    # StoriesDataModule(small) -> one val batch
    from hardcoregenai.pipeline.stories_data_module import StoriesDataModule
    sdm = StoriesDataModule(batch_size=8, seq_len=64, data_size="small")
    sdm.num_proc = 1
    sdm.prepare_data()
    sdm.setup("fit")
    val_loader = sdm.val_dataloader()
    batch = next(iter(val_loader))
    tokens = batch["tokens"]
    
    ce = evaluate_ce(model, tokens)
    MAX_ACCEPTABLE_CE = 3.0 # Adjust to your hardware, model size, and avail. training time
    assert ce < MAX_ACCEPTABLE_CE, f"CE unexpectedly large ({MAX_ACCEPTABLE_CE=})"


@pytest.mark.skip("Test is WIP")
@pytest.mark.lab("unit1.lab2")
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize(
    "prompt, expected, unexpected",
    [
        ("He said to",           [" him", " her", " them"],    [" he", " said", " He"]),
        ("She looked at the",    [" door", " boy", " girl"],   [" wish", " and", "."]),
        ("They walked into the", [" big", " door", " forest"], [" what", " Dad", " the"]),
    ],
)
def test_ckpt_prompt_ranking(prompt, expected, unexpected):
    # TODO Test is still WIP
    ckpt = find_checkpoint()
    if ckpt is None:
        pytest.skip("No checkpoint found. To run this test, you must first train the model: use app/tell_stories_with_neural_ngram.py")

    model = load_ckpt_cpu(ckpt, NeuralNgram)
    model.eval()

    from hardcoregenai.pipeline.bpe_tokenizer import HcgaiTokenizer
    tok = HcgaiTokenizer

    def ids(s: str):
        return list(map(int, tok.tokenize(s)))

    ctx = ids(prompt)
    x = torch.tensor([ctx[-64:]], dtype=torch.long)  # (1, <=64)

    with torch.inference_mode():
        logits = model(x[:, :-1])            # (1, T-1, V)
    last = logits[0, -1, :]                  # (V,)
    probs = torch.softmax(last, dim=-1)      # compare in probability space

    exp_ids = [ids(w)[0] for w in expected]
    unx_ids = [ids(w)[0] for w in unexpected]
    print(f'{ids=} {probs=}')

    exp_mean = float(probs[exp_ids].min().item())
    unx_max  = float(probs[unx_ids].max().item())
    MIN_PROB_MULTIPLE = 10.0

    # TODO Clarify msg
    assert exp_mean > unx_max * MIN_PROB_MULTIPLE, f"Expected min prob {exp_mean:.3f} not > unexpected max {unx_max:.3f} * {MIN_PROB_MULTIPLE=}"
