"""Unit tests for the reference Transformer's ablation toggles.

The model under test mirrors src/ft4/models/transformer.py but with the
ft4.pipeline imports (StoriesDataModule, Ft4Tokenizer) stripped — those
require real data infrastructure. Model logic is identical.

Defining the test fixture inline (rather than importing from src/) keeps
the test self-contained: no fixture module in the production package, no
"why is this file in src/?" question.

Verifies:
  - Each toggle constructs without error in both settings.
  - Forward returns the expected shape under each setting.
  - state_dict_hash is unchanged by either toggle (both are non-parametric).
  - save_hyperparameters captures both toggles.
  - Each toggle, when enabled, observably changes the model's output.
"""
import math

import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from mlops.hset_state import hash_state_dict
from ft4.models.mlp import MlpBlock
from ft4.models.attention import Attention


# Must match the constant in src/ft4/models/transformer.py.
LOGIT_SOFTCAP = 40.0


class _TransformerLayer(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4):
        super().__init__()
        self.attn = Attention(dim)
        self.mlp = MlpBlock(dim, expansion_factor=expansion_factor)

    def forward(self, x):
        return self.mlp(self.attn(x))


class _Transformer(L.LightningModule):
    """Stripped-down clone of ft4.models.transformer.Transformer. Same
    logic, no pipeline imports. Test-only."""

    def __init__(self, vocab_size: int, dim: int, expansion_factor: int = 4, depth: int = 8,
                 padding_idx: int = 0, use_pos_enc: bool = False, use_logit_softcap: bool = False,
                 lr: float = 3e-4, weight_decay: float = 1e-2, warmup_steps: int = 3000):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.unembed = nn.Linear(dim, vocab_size)
        self.unembed.weight = self.embed.weight
        self.layers = nn.Sequential(*[_TransformerLayer(dim, expansion_factor) for _ in range(depth)])
        pos_enc = self._positional_encoding().to(self.embed.weight.dtype) / math.sqrt(dim)
        self.register_buffer('positional_encoding', pos_enc, persistent=False)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self._token2vec(tokens)
        assert x.shape == (B, L, self.dim)
        y = self.layers(x)
        return self._y2logits(y)

    def _token2vec(self, tokens):
        x = self.embed(tokens)
        seq_len = x.shape[1]
        if self.hparams.use_pos_enc:  # type: ignore
            x = x + self.positional_encoding[:seq_len, :]  # type: ignore
        return x

    def _y2logits(self, y):
        logits = self.unembed(y)
        if self.hparams.use_logit_softcap:  # type: ignore
            logits = LOGIT_SOFTCAP * (logits / LOGIT_SOFTCAP).tanh()
        return logits

    def _positional_encoding(self, max_seq_len: int = 2048):
        N = 10_000
        assert self.dim % 2 == 0
        pos_in_seq = torch.arange(max_seq_len)
        dim_idx = torch.arange(self.dim)
        angle_rates = N ** -(dim_idx / self.dim)
        angles = torch.einsum('l,d->ld', pos_in_seq, angle_rates)
        pos_enc = torch.empty_like(angles)
        pos_enc[:, 0::2] = torch.sin(angles)[:, 0::2]
        pos_enc[:, 1::2] = torch.cos(angles)[:, 0::2]
        return pos_enc

    def training_step(self, batch):
        tokens = batch['tokens']
        x, y = tokens[..., :-1], tokens[..., 1:]
        logits = self(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=self.padding_idx)
        self.log('train_ce', loss)
        return loss

    def validation_step(self, batch):
        tokens = batch['tokens']
        x, y = tokens[..., :-1], tokens[..., 1:]
        logits = self(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=self.padding_idx)
        self.log('val_ce', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def _tiny(**kw) -> _Transformer:
    return _Transformer(vocab_size=32, dim=16, depth=2, **kw)


# ── Transformer constructs and forwards ───────────────────────────────────

def test_transformer_default_constructs_and_forwards():
    t = _tiny()
    tokens = torch.randint(1, 32, (2, 8))
    out = t(tokens)
    assert out.shape == (2, 8, 32)


def test_transformer_with_each_toggle_on():
    for flag in ("use_pos_enc", "use_logit_softcap"):
        t = _tiny(**{flag: True})
        tokens = torch.randint(1, 32, (2, 8))
        assert t(tokens).shape == (2, 8, 32), f"forward failed with {flag}=True"


def test_transformer_with_both_toggles_on():
    t = _tiny(use_pos_enc=True, use_logit_softcap=True)
    tokens = torch.randint(1, 32, (2, 8))
    assert t(tokens).shape == (2, 8, 32)


# ── save_hyperparameters captures toggles ─────────────────────────────────

def test_hparams_captures_both_toggles():
    t = _tiny(use_pos_enc=True, use_logit_softcap=True)
    assert t.hparams.use_pos_enc is True
    assert t.hparams.use_logit_softcap is True


def test_hparams_defaults_are_false():
    """Course-standard configuration: both toggles default off."""
    t = _tiny()
    assert t.hparams.use_pos_enc is False
    assert t.hparams.use_logit_softcap is False


# ── state_dict_hash behavior ──────────────────────────────────────────────

def test_state_dict_hash_unchanged_by_use_pos_enc():
    """PE buffer is non-persistent; toggling does not change state_dict.
    Verified against ft4's hash_state_dict (the same function the CLI uses)."""
    torch.manual_seed(0)
    t1 = _tiny()
    torch.manual_seed(0)
    t2 = _tiny(use_pos_enc=True)
    assert hash_state_dict(t1) == hash_state_dict(t2)


def test_state_dict_hash_unchanged_by_use_logit_softcap():
    """Softcap is pure forward-pass logic; no parameters."""
    torch.manual_seed(0)
    t1 = _tiny()
    torch.manual_seed(0)
    t2 = _tiny(use_logit_softcap=True)
    assert hash_state_dict(t1) == hash_state_dict(t2)


# ── Toggles observably change behavior ────────────────────────────────────

def test_softcap_changes_output_when_enabled():
    """With softcap on, large logits get squashed via tanh; bounded by cap."""
    torch.manual_seed(0)
    t_off = _tiny(use_logit_softcap=False)
    torch.manual_seed(0)
    t_on = _tiny(use_logit_softcap=True)
    tokens = torch.randint(1, 32, (2, 8))
    with torch.no_grad():
        out_off = t_off(tokens)
        out_on = t_on(tokens)
    assert not torch.allclose(out_off, out_on)
    assert out_on.abs().max().item() < LOGIT_SOFTCAP


def test_pos_enc_changes_output_when_enabled():
    """With PE on, the positional encoding is added to the embedding."""
    torch.manual_seed(0)
    t_off = _tiny(use_pos_enc=False)
    torch.manual_seed(0)
    t_on = _tiny(use_pos_enc=True)
    tokens = torch.randint(1, 32, (2, 8))
    with torch.no_grad():
        out_off = t_off(tokens)
        out_on = t_on(tokens)
    assert not torch.allclose(out_off, out_on)


# ── Cross-toggle independence ─────────────────────────────────────────────

def test_pos_enc_and_softcap_can_combine():
    """Both toggles enabled simultaneously: forward works, softcap still bounds."""
    t = _tiny(use_pos_enc=True, use_logit_softcap=True)
    tokens = torch.randint(1, 32, (2, 8))
    out = t(tokens)
    assert out.shape == (2, 8, 32)
    assert out.abs().max().item() < LOGIT_SOFTCAP
