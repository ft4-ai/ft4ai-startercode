
import pytest
import math
from typing import Tuple
from collections import defaultdict, Counter

import torch

from ft4.models.empirical_model import EmpiricalModel

PAD_ID = 0
VOCAB_SIZE = 6

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_forward_shape_and_finiteness(n, corpus):
    train_tokens, _ = corpus
    model = make_model(n=n)
    train_once(model, train_tokens)

    logits = model(train_tokens)  # [B, L, V]

    assert logits.shape == (train_tokens.shape[0], train_tokens.shape[1], VOCAB_SIZE)

    # No row should be all -inf after model's fallback.
    is_all_neginf = torch.isneginf(logits).all(dim=-1)
    assert not bool(is_all_neginf.any())

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_argmax_matches_empirical_mode_for_seen_contexts(n, corpus):
    train_tokens, _ = corpus
    model = make_model(n=n)
    train_once(model, train_tokens)

    # Build empirical table only from seen, non-padded contexts.
    table = build_ngram_counts(train_tokens.tolist(), n=n)

    logits = model(train_tokens)
    B, L, V = logits.shape
    cw = max(0, n-1)

    # Check a few positions safely inside the sequence (i >= cw).
    checks = 0
    for b in range(B):
        for i in range(cw, L):
            ctx = tuple(train_tokens[b, i-cw:i].tolist()) if cw > 0 else tuple()
            if ctx not in table or len(table[ctx]) == 0:
                continue
            # Model top-1
            row = logits[b, i-1, :].clone()  # logits for next-token at position i
            row[PAD_ID] = float("-inf")
            top_idx = int(torch.argmax(row).item())
            # Empirical mode (break ties by accepting any of the tied maxima)
            cnts = table[ctx]
            max_c = max(cnts.values())
            tied = {k for k, v in cnts.items() if v == max_c}
            assert top_idx in tied
            checks += 1
    # Ensure we actually checked some positions
    assert checks >= 3

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_unseen_prefix_gives_uniform_logits(n, corpus):
    train_tokens, _ = corpus
    model = make_model(n=n)
    train_once(model, train_tokens)

    # Craft a batch where contexts are all token '5' (unused in training) to force unseen prefixes.
    B = 2
    L = 6
    unseen_tok = 5  # not used in training corpus
    x = torch.full((B, L), unseen_tok, dtype=torch.long)
    logits = model(x) # [B, L, V]

    # Expect uniform rows throughout (or at least at later positions)
    # Check last 3 positions to be robust.
    for b in range(B):
        for i in range(L-3, L):
            assert row_is_uniform(logits[b, i, :])

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("vocab_size", [VOCAB_SIZE, 16_384])
@pytest.mark.filterwarnings("ignore:You are trying to `self\\.log\\(\\)`:UserWarning")
def test_validation_ce_is_reasonable(n, vocab_size, corpus):
    train_tokens, val_tokens = corpus
    model = make_model(n=n, vocab_size=vocab_size)
    train_once(model, train_tokens)

    model.on_validation_start()
    ce = model.validation_step({"tokens": val_tokens})
    model.on_validation_end()
    # Lightning may return a Tensor or a float; normalize
    ce_val = float(ce.detach().cpu().item() if isinstance(ce, torch.Tensor) else ce)
    assert math.isfinite(ce_val)
    # Upper bound near log(V); allow slack
    assert 0.0 <= ce_val <= math.log(VOCAB_SIZE) + 1.0

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [2, 3, 4])
def test_batch_consistency_vs_single_items(n):
    # Two distinct sequences of equal length
    s1 = torch.tensor([[1,2,1,3,1,2,1,4]], dtype=torch.long)  # [1,L]
    s2 = torch.tensor([[1,2,1,3,2,1,4,1]], dtype=torch.long)  # [1,L]
    batch = torch.cat([s1, s2], dim=0)  # [2,L]

    model = make_model(n=n)
    train_once(model, batch)

    # Forward on the full batch
    logits_batch = model(batch)                    # [2,L,V]
    # Forward separately, then stack — these must match the batch rows
    logits_sep = torch.cat([model(s1), model(s2)], dim=0)
    torch.testing.assert_close(logits_batch, logits_sep, atol=0, rtol=0)

@pytest.fixture(scope="session")
def corpus() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Training and validation corpora with **fixed-length** sequences (no padding).
    Token ids use {1,2,3,4}; id 5 remains unused to force unseen-prefix tests.
    """
    # Training batch (B=2), both length 8
    s1 = [1, 2, 1, 3, 1, 2, 1, 4]           # len 8
    s2 = [1, 2, 1, 3, 2, 1, 4, 1]           # len 8 (made equal-length)
    train = torch.tensor([s1, s2], dtype=torch.long)
    # Validation batch (B=2), both length 9
    v1 = [1, 2, 1, 3, 1, 2, 1, 4, 3]        # len 9
    v2 = [1, 2, 1, 3, 2, 1, 4, 1, 2]        # len 9 (made equal-length)
    val = torch.tensor([v1, v2], dtype=torch.long)
    return train, val

def make_model(n: int, vocab_size=VOCAB_SIZE, **kwargs):
    return EmpiricalModel(n=n, vocab_size=vocab_size, **kwargs)

def train_once(model, train_tokens: torch.Tensor):
    # Single counting pass; minimal Lightning interaction.
    model.training = True
    model.training_step({"tokens": train_tokens}, batch_idx=0)
    model.eval()
    return model

def build_ngram_counts(seqs, n: int):
    """
    Pure-Python n-gram counter for *seen* (non-padded) contexts only.
    Avoids padding assumptions by using only positions i >= cw inside the sequence.
    Returns: dict mapping tuple(context) -> Counter(next_token)
    """
    cw = max(0, n-1)
    table = defaultdict(Counter)
    for seq in seqs:
        L = len(seq)
        for i in range(cw, L-1):
            ctx = tuple(seq[i-cw:i]) if cw > 0 else tuple()
            nxt = seq[i]
            # Skip any window using PAD in context or as target
            if (cw > 0 and any(t == PAD_ID for t in ctx)) or nxt == PAD_ID:
                continue
            table[ctx][nxt] += 1
    return table

def row_is_uniform(row: torch.Tensor, atol=1e-7) -> bool:
    return bool(torch.isfinite(row).all()) and float((row.max() - row.min()).abs()) <= atol


######################

# The test below checks that we can save and restore checkpoints

import os
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

class TinyTokDataset(Dataset):
    def __init__(self, seqs):
        self._seqs = seqs
    def __len__(self): return len(self._seqs)
    def __getitem__(self, idx): return {"tokens": self._seqs[idx]}

@pytest.mark.lab("unit1.lab1")
@pytest.mark.parametrize("n", [3])  # one representative n for checkpoint test
@pytest.mark.filterwarnings("ignore::Warning")
def test_trainer_checkpoint_roundtrip(tmp_path: Path, n, corpus):
    train_tokens, val_tokens = corpus

    model = make_model(n=n, min_distinct_suffixes=1)

    # One-batch loaders
    train_ds = TinyTokDataset([train_tokens[0]])
    val_ds   = TinyTokDataset([val_tokens[0]])
    train_loader = DataLoader(train_ds, batch_size=1)
    val_loader   = DataLoader(val_ds, batch_size=1)

    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), save_last=True, save_top_k=1, monitor=None)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        logger=False,
        enable_model_summary=False,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Resolve checkpoint path
    ckpt_path = ckpt_cb.last_model_path or ckpt_cb.best_model_path
    assert ckpt_path and os.path.exists(ckpt_path)

    # Load via Lightning's API (restores hyperparams and on_load_checkpoint)
    loaded = EmpiricalModel.load_from_checkpoint(ckpt_path)

    # Simple behavioral check: argmax at a known position matches
    x = train_tokens[:1]  # one seq
    logits1 = model(x)
    logits2 = loaded(x)
    torch.testing.assert_close(torch.argmax(logits1, dim=-1), torch.argmax(logits2, dim=-1))

