"""One sanity test for `mlops.prepare.prepare_run`.

The existing test_run* suite exhaustively covers Phase 0 behavior via the
`run()` entry point. This test guards the public contract of `prepare_run`
itself: that it returns a populated PreparedRun for a write op on a fresh
runs_root, with the expected v001/h001 layout created.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from mlops.hset_state import Ft4Args
from mlops.prepare import PreparedRun, _ensure_session_finalization, prepare_run
from mlops.session_finalization import SessionFinalizationCallback


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
            return nn.functional.cross_entropy(self(x), y)
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)


    class TinyDataModule(L.LightningDataModule):
        def setup(self, stage=None):
            torch.manual_seed(0)
            x = torch.randn(16, 4)
            y = torch.randint(0, 2, (16,))
            self.train_ds = TensorDataset(x, y)
        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=4)
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
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny.py"
    model_file.write_text(_TINY_MODEL_SRC)
    (tmp_path / "tiny.yaml").write_text(_MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(_DEFAULTS_YAML)
    # Point the cascade at our tmp defaults instead of the shipped layers,
    # so the test environment is fully self-contained.
    import mlops.main as main_mod
    monkeypatch.setattr(
        main_mod, "_default_cascade_paths",
        lambda _mf: [defaults, tmp_path / "tiny.yaml"],
    )
    sys.modules.pop("tiny", None)
    yield model_file, tmp_path / "runs"
    sys.modules.pop("tiny", None)


def test_prepare_run_populates_everything(env):
    model_file, runs_root = env
    args = Ft4Args(subcommand="train", model_file=model_file)
    p = prepare_run(args, runs_root=runs_root)

    assert isinstance(p, PreparedRun)
    # Phase 0 on a fresh runs_root for a write op creates v001/h001.
    assert p.version.name == "v001"
    assert p.hset.name == "h001"
    assert p.ckpt_path is None
    assert p.sessions == []
    # All three signals computed.
    assert p.signals.state_dict_hash and p.signals.model_file_hash and p.signals.config_hash
    # Trainer + model attached (same objects as on the cli).
    assert p.model is p.cli.model
    assert p.trainer is p.cli.trainer
    # Trainer summary captured max_epochs from the cascade.
    assert p.trainer_summary.max_epochs == 1


class _FakeTrainer:
    def __init__(self, callbacks):
        self.callbacks = list(callbacks)


def test_ensure_session_finalization_noop_when_present():
    cb = SessionFinalizationCallback()
    trainer = _FakeTrainer([cb])
    _ensure_session_finalization(trainer)
    assert trainer.callbacks == [cb]


def test_ensure_session_finalization_appends_when_missing(capsys):
    trainer = _FakeTrainer([])
    _ensure_session_finalization(trainer)
    assert len(trainer.callbacks) == 1
    assert isinstance(trainer.callbacks[0], SessionFinalizationCallback)
    err = capsys.readouterr().err
    assert "SessionFinalizationCallback" in err
