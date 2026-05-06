"""
Generic helpers for next-token language-model tests.
Small, black-box utilities designed to work across different model classes.
"""
import glob
import os
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
    *,
    env_var: str = "CKPT_NEURAL_NGRAM",
    glob_pattern: str = "./checkpoints/neural_ngram/**/*.ckpt",
) -> Optional[str]:
    """Checkpoint policy: env var first, else newest .ckpt by mtime under glob_pattern. Returns path or None."""
    p = os.getenv(env_var)
    if p and os.path.exists(p):
        return p
    candidates = glob.glob(glob_pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda q: os.path.getmtime(os.path.realpath(q)))
    return candidates[-1]


def load_ckpt_cpu(ckpt_path: str, cls: type):
    """Load a Lightning checkpoint on CPU by default for robustness."""
    return cls.load_from_checkpoint(ckpt_path, map_location="cpu")
