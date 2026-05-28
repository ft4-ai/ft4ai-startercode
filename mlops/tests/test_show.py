"""Integration tests for `ft4 show`.

Tests the read-only briefing: after running `ft4 run` once on a fixture
model, `ft4 show` reproduces the same briefing without training.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from mlops.hset_state import Ft4Args, Ft4UserError
from mlops.run import run
from mlops.show import run_show


_TINY_MODEL_SRC = textwrap.dedent("""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import lightning as L

    class TinyModel(L.LightningModule):
        def __init__(self, hidden: int = 8):
            super().__init__()
            self.save_hyperparameters()
            self.fc = nn.Linear(4, hidden)
            self.out = nn.Linear(hidden, 2)
        def forward(self, x):
            return self.out(self.fc(x))
        def training_step(self, batch, _):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss)
            return loss
        def validation_step(self, batch, _):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("val_loss", loss)
            return loss
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)


    class TinyDataModule(L.LightningDataModule):
        def setup(self, stage=None):
            torch.manual_seed(0)
            x = torch.randn(16, 4)
            y = torch.randint(0, 2, (16,))
            self.train_ds = TensorDataset(x[:12], y[:12])
            self.val_ds = TensorDataset(x[12:], y[12:])
        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=4)
        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=4)
""")


_MODEL_YAML = textwrap.dedent("""
    data:
      class_path: tiny.TinyDataModule
""")


_DEFAULTS_YAML = textwrap.dedent("""
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


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny.py"
    model_file.write_text(_TINY_MODEL_SRC)
    (tmp_path / "tiny.yaml").write_text(_MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(_DEFAULTS_YAML)
    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])
    sys.modules.pop("tiny", None)
    yield model_file, tmp_path / "runs"
    sys.modules.pop("tiny", None)


def test_show_errors_when_no_runs(env):
    """Without any prior `ft4 run`, `ft4 show` errors via resolve_version."""
    model_file, runs_root = env
    args = Ft4Args(subcommand="show", model_file=model_file)
    with pytest.raises(Ft4UserError):
        run_show(args, runs_root=runs_root)


def test_show_prints_briefing_after_run(env, capsys, monkeypatch):
    model_file, runs_root = env
    # Train once so v001/h001 exists with a session record.
    run(Ft4Args(subcommand="train", model_file=model_file), runs_root=runs_root)
    capsys.readouterr()  # discard training output

    # Force non-TTY path so output is captured as plain text.
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    code = run_show(
        Ft4Args(subcommand="show", model_file=model_file),
        runs_root=runs_root,
    )
    assert code == 0
    out = capsys.readouterr().out
    # Briefing identifies the version/hset and points at known paths.
    assert "v001" in out
    assert "h001" in out
    assert "runs/tiny/v001/h001" in out


def test_show_works_on_old_version_after_model_file_change(env, capsys, monkeypatch):
    """Regression: editing the model file must not break `ft4 show --version vNNN`."""
    model_file, runs_root = env
    run(Ft4Args(subcommand="train", model_file=model_file), runs_root=runs_root)
    capsys.readouterr()

    # Edit the model file so its hash no longer matches v001's stored hash.
    model_file.write_text(model_file.read_text() + "\n# touched\n")

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    code = run_show(
        Ft4Args(subcommand="show", model_file=model_file, version="v001"),
        runs_root=runs_root,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "v001" in out
    assert "h001" in out
