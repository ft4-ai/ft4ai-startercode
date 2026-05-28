"""Integration test: ft4 smart-flow against the reference Transformer toggles.

Verifies, end-to-end through the CLI, that flipping either toggle (non-
parametric, forward-pass only) forks a new HSET within the same version.
Both `use_pos_enc` and `use_logit_softcap` follow this pattern.

The version-fork mechanism (state_dict shape change) is exercised by
mlops/tests/test_hset_state.py::test_resolve_version_changed_arch_forks;
it fires naturally any time `dim` or `depth` changes.

The MODEL_SRC string below contains a fully self-contained Transformer —
no inheritance from anything in ft4.models — so the test doesn't depend
on the production transformer.py being importable (which it isn't in CI
without the StoriesDataModule / Ft4Tokenizer infrastructure). The logic
mirrors ft4.models.transformer.Transformer.
"""
import json
import sys
import textwrap
from pathlib import Path

import pytest

from mlops.run import run
from mlops.hset_state import Ft4Args


MODEL_SRC = textwrap.dedent("""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    import lightning as L

    from ft4.models.mlp import MlpBlock
    from ft4.models.attention import Attention

    LOGIT_SOFTCAP = 40.0


    class _TransformerLayer(nn.Module):
        def __init__(self, dim, expansion_factor=4):
            super().__init__()
            self.attn = Attention(dim)
            self.mlp = MlpBlock(dim, expansion_factor=expansion_factor)

        def forward(self, x):
            return self.mlp(self.attn(x))


    class Transformer(L.LightningModule):
        def __init__(self, vocab_size, dim, expansion_factor=4, depth=8, padding_idx=0,
                     use_pos_enc=False, use_logit_softcap=False,
                     lr=3e-4, weight_decay=1e-2, warmup_steps=3000):
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
            x = self.embed(tokens)
            if self.hparams.use_pos_enc:
                x = x + self.positional_encoding[:L, :]
            y = self.layers(x)
            logits = self.unembed(y)
            if self.hparams.use_logit_softcap:
                logits = LOGIT_SOFTCAP * (logits / LOGIT_SOFTCAP).tanh()
            return logits

        def _positional_encoding(self, max_seq_len=2048):
            N = 10_000
            assert self.dim % 2 == 0
            pos_in_seq = torch.arange(max_seq_len)
            dim_idx = torch.arange(self.dim)
            angle_rates = N ** -(dim_idx/self.dim)
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


    class _TokensDataset(Dataset):
        def __init__(self, tokens):
            self.tokens = tokens
        def __len__(self):
            return len(self.tokens)
        def __getitem__(self, i):
            return {'tokens': self.tokens[i]}


    class SyntheticLMDataModule(L.LightningDataModule):
        def __init__(self, vocab_size=32, seq_len=8, batch_size=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.batch_size = batch_size

        def setup(self, stage=None):
            torch.manual_seed(0)
            tokens = torch.randint(1, self.vocab_size, (24, self.seq_len + 1))
            self.train_ds = _TokensDataset(tokens[:16])
            self.val_ds = _TokensDataset(tokens[16:])

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size)
""")


DEFAULTS_YAML = textwrap.dedent("""
    trainer:
      max_epochs: 1
      enable_progress_bar: false
      enable_model_summary: false
      logger:
        class_path: lightning.pytorch.loggers.CSVLogger
        init_args:
          save_dir: "."
          name: ""
          version: ""
      callbacks:
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            dirpath: "."
            filename: "{epoch}"
            save_last: true
            save_top_k: -1
            monitor: null
        - class_path: mlops.session_finalization.SessionFinalizationCallback
""")


MODEL_YAML = textwrap.dedent("""
    model:
      vocab_size: 32
      dim: 16
      depth: 2
      expansion_factor: 4

    data:
      class_path: transformer_under_test.SyntheticLMDataModule
      init_args:
        vocab_size: 32
        seq_len: 8
        batch_size: 4
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "transformer_under_test.py"
    model_file.write_text(MODEL_SRC)
    (tmp_path / "transformer_under_test.yaml").write_text(MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])
    sys.modules.pop("transformer_under_test", None)
    yield SimpleEnv(model_file=model_file, runs_root=tmp_path / "runs")
    sys.modules.pop("transformer_under_test", None)


class SimpleEnv:
    def __init__(self, model_file: Path, runs_root: Path):
        self.model_file = model_file
        self.runs_root = runs_root


def _args(env: SimpleEnv, lightning_args=None) -> Ft4Args:
    return Ft4Args(
        subcommand="train",
        model_file=env.model_file,
        lightning_args=lightning_args or [],
    )


# ── use_pos_enc: hset fork ───────────────────────────────────────────────

def test_use_pos_enc_forks_a_new_hset_within_same_version(env):
    """Flipping use_pos_enc keeps state_dict_hash constant (PE is a
    non-persistent buffer), so ft4 keeps the same version and forks a
    new hset."""
    run(_args(env, ["--model.use_pos_enc=false"]), runs_root=env.runs_root)
    run(_args(env, ["--model.use_pos_enc=true"]), runs_root=env.runs_root)

    model_root = env.runs_root / "transformer_under_test"
    versions = sorted(p.name for p in model_root.iterdir() if p.is_dir())
    assert versions == ["v001"], (
        f"expected single version; got {versions} — use_pos_enc should NOT "
        f"fork a new version since PE is a non-persistent buffer"
    )

    hsets = sorted(p.name for p in (model_root / "v001").iterdir()
                    if p.is_dir() and p.name.startswith("h"))
    assert hsets == ["h001", "h002"], (
        f"expected two hsets within v001; got {hsets}"
    )

    for t in hsets:
        assert (model_root / "v001" / t / "checkpoints" / "last.ckpt").is_file()

    t1_json = json.loads((model_root / "v001" / "h001" / "hset.json").read_text())
    t2_json = json.loads((model_root / "v001" / "h002" / "hset.json").read_text())
    assert t1_json["config_hash"] != t2_json["config_hash"]


# ── use_logit_softcap: same hset-fork story ──────────────────────────────

def test_use_logit_softcap_forks_a_new_hset_within_same_version(env):
    """Same story as use_pos_enc: non-parametric forward-only change."""
    run(_args(env, ["--model.use_logit_softcap=false"]), runs_root=env.runs_root)
    run(_args(env, ["--model.use_logit_softcap=true"]), runs_root=env.runs_root)

    model_root = env.runs_root / "transformer_under_test"
    versions = [p.name for p in model_root.iterdir() if p.is_dir()]
    assert versions == ["v001"]
    hsets = sorted(p.name for p in (model_root / "v001").iterdir()
                    if p.is_dir() and p.name.startswith("h"))
    assert hsets == ["h001", "h002"]


# ── Both toggles simultaneously still single version, multi hset ────────

def test_both_toggles_combine_in_hset_fork(env):
    """Three combinations: (off,off), (on,off), (on,on) all in same version.
    Confirms toggles don't interact with state_dict and hsets accumulate."""
    run(_args(env, ["--model.use_pos_enc=false", "--model.use_logit_softcap=false"]),
        runs_root=env.runs_root)
    run(_args(env, ["--model.use_pos_enc=true",  "--model.use_logit_softcap=false"]),
        runs_root=env.runs_root)
    run(_args(env, ["--model.use_pos_enc=true",  "--model.use_logit_softcap=true"]),
        runs_root=env.runs_root)

    model_root = env.runs_root / "transformer_under_test"
    versions = sorted(p.name for p in model_root.iterdir() if p.is_dir())
    assert versions == ["v001"]

    hsets = sorted(p.name for p in (model_root / "v001").iterdir()
                    if p.is_dir() and p.name.startswith("h"))
    assert hsets == ["h001", "h002", "h003"]
