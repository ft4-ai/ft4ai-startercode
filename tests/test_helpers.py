"""
Generic helpers for next-token language-model tests.
Small, black-box utilities designed to work across different model classes.
"""
import glob
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor


# -------- Core primitives --------

def set_seed(seed: int = 1234) -> None:
    """Seed torch RNG for reproducible tests."""
    torch.manual_seed(seed)


def make_tokens(
    batch_size: int,
    total_len: int,
    vocab_size: int,
    *,
    avoid_id: int | None = None,
    seed: int = 1234,
) -> Tensor:
    """Create synthetic LongTensor of shape (B, total_len) with IDs in [0, vocab_size-1].
    If avoid_id is set, that id will not appear.
    """
    assert vocab_size >= 2, "vocab_size must be >= 2"
    g = torch.Generator().manual_seed(seed)
    hi = vocab_size - (1 if avoid_id is not None else 0)
    t = torch.randint(low=0, high=hi, size=(batch_size, total_len), dtype=torch.long, generator=g)
    if avoid_id is not None:
        # Map values >= avoid_id to +1 to skip the avoid_id
        t = torch.where(t >= avoid_id, t + 1, t)
    return t


def xy_from_tokens(tokens: Tensor) -> Tuple[Tensor, Tensor]:
    """Split tokens (B, L+1) into x=(B, L), y=(B, L) for next-token prediction."""
    assert tokens.ndim == 2 and tokens.dtype == torch.long, "tokens must be LongTensor of shape (B, L+1)"
    return tokens[:, :-1], tokens[:, 1:]


def ce_from_logits(logits: Tensor, y: Tensor, *, ignore_index: int | None = None) -> Tensor:
    """Cross-entropy over flattened time: logits (B, L, V), y (B, L)."""
    logits_ = logits.flatten(0, 1)
    y_ = y.flatten()
    return F.cross_entropy(logits_, y_, ignore_index=ignore_index if ignore_index is not None else -100)


# -------- Convenience wrappers --------

def evaluate_ce(
    model: object,
    tokens: Tensor,
    *,
    ignore_index: int | None = None,
    forward_fn: Callable[[object, Tensor], Tensor] | None = None,
) -> float:
    """End-to-end CE on tokens (B, L+1) via model forward.
    forward_fn: optional adapter, default calls model(x).
    Returns a Python float.
    """
    x, y = xy_from_tokens(tokens)
    if forward_fn is None:
        logits = model(x)  # type: ignore[attr-defined]
    else:
        logits = forward_fn(model, x)
    ce = ce_from_logits(
        logits,
        y,
        ignore_index=ignore_index if ignore_index is not None else getattr(model, "padding_idx", None),
    )
    return float(ce.detach().cpu().item())


def make_trainer(
    *,
    max_steps: int = 200,
    limit_train_batches: int = 1,
    limit_val_batches: int = 1,
    accelerator: str = "auto",
    precision: str = "32-true",
    deterministic: bool = False,
) -> L.Trainer:
    """A standard Lightning Trainer for one-batch tests; logging/ckpt disabled."""
    return L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision, #type:ignore
        max_steps=max_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=deterministic,
    )


def find_checkpoint(
    model_stem: str,
    *,
    runs_root: str = "./runs",
    env_var: str | None = None,
) -> Optional[str]:
    """Locate a trained checkpoint for `model_stem` (e.g. 'neural_ngram').

    Resolution order:
      1. Env var `CKPT_<MODEL_STEM_UPPER>` (or override via `env_var=`) if set and the path exists.
      2. Newest `last.ckpt` under {runs_root}/{model_stem}/**, by mtime of the
         resolved (post-symlink) target.

    Returns the path string, or None if no checkpoint is found.
    """
    env_name = env_var or f"CKPT_{model_stem.upper()}"
    p = os.getenv(env_name)
    if p and os.path.exists(p):
        return p
    pattern = os.path.join(runs_root, model_stem, "**", "last.ckpt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda q: os.path.getmtime(os.path.realpath(q)))
    return candidates[-1]


def load_ckpt_cpu(ckpt_path: str, cls: type):
    """Load a Lightning checkpoint on CPU by default for robustness."""
    return cls.load_from_checkpoint(ckpt_path, map_location="cpu")


def stories_val_batch(*, batch_size: int = 8, seq_len: int = 64, data_size: str = "small") -> Tensor:
    """Return the first val batch from StoriesDataModule as a (B, L+1) LongTensor."""
    from ft4.pipeline.stories_data_module import StoriesDataModule
    sdm = StoriesDataModule(batch_size=batch_size, seq_len=seq_len, data_size=data_size) #type:ignore
    sdm.num_proc = 1
    sdm.prepare_data()
    sdm.setup("fit")
    return next(iter(sdm.val_dataloader()))["tokens"]


# -------- Architecture tests (no checkpoint required) --------

def assert_eval_deterministic(model, x: Tensor) -> None:
    """Verify model is deterministic in eval mode.

    Runs model(x) twice in eval() under no_grad() and asserts identical outputs.
    Catches stray functional dropout (or other stochastic ops) left enabled in inference.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        if not torch.equal(out1, out2):
            raise AssertionError(
                "Model produced different outputs for the same input in eval mode. "
                "Check for functional dropout (or other stochastic ops) with hard-coded probabilities."
            )
    finally:
        if was_training:
            model.train()


def assert_causal(model, x: Tensor, *, positions: list[int] | None = None) -> None:
    """Verify that logits at position t depend only on tokens at positions <= t.

    For each k in `positions` (default: two sample positions), perturb x[:, k], re-run, and
    assert logits at positions 0..k-1 are bit-identical. Also asserts the perturbation actually
    affected *something* at positions >= k, so the test isn't silently inert.

    x: (B, L) LongTensor with L >= 4.
    """
    assert x.ndim == 2 and x.dtype == torch.long
    B, L = x.shape
    assert L >= 4, f"need L >= 4 to test causality, got L={L}"

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model(x)
            V = base.shape[-1]
            if positions is None:
                positions = [L // 3, 2 * L // 3]
            for k in positions:
                assert 0 < k < L, f"position {k} out of range for L={L}"
                x2 = x.clone()
                x2[:, k] = (x[:, k] + 1) % V  # guaranteed-different token
                perturbed = model(x2)
                if not torch.equal(base[:, :k, :], perturbed[:, :k, :]):
                    raise AssertionError(
                        f"Causality violated: perturbing token at position {k} "
                        f"changed logits at earlier positions (0..{k-1})."
                    )
                if torch.equal(base[:, k:, :], perturbed[:, k:, :]):
                    raise AssertionError(
                        f"Test inert: perturbing token at position {k} had no effect on "
                        f"any later logits, so this run cannot discriminate causal from acausal."
                    )
    finally:
        if was_training:
            model.train()


# TODO: padding invariance test.
# Prepending pad tokens to x should not change logits at non-pad positions (up to a shift).
# Deferred until padding handling across models is audited — a test failure here today could
# point at padding code, not at the model under test.

