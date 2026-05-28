"""End-to-end smoke test for the Lightning integration.

Validates the load-bearing seam without the smart flow:
  discover_model_class
    → build_lightning_cli (cascade YAML, run=False, save_config_callback=None)
      → inject_paths (mutate ModelCheckpoint.dirpath + logger._save_dir)
        → persist_config (write resolved config.yaml)
          → trainer.fit (one epoch on synthetic data)

This is the cheapest way to discover Lightning-seam surprises before more
code accumulates on top of unverified assumptions.

The TinyModel + TinyDataModule defined inline together exercise the realistic
case where a user's <model>.yaml references the datamodule by natural stem
(e.g., `class_path: tiny.TinyDataModule`) and jsonargparse must resolve it
through sys.modules.
"""
import sys
import textwrap
from pathlib import Path

import pytest

from mlops.cumulative_metrics import rebuild_cumulative_metrics
from mlops.main import build_lightning_cli, discover_model_class
from mlops.hset_state import (
    Ft4Args,
    create_session_dir,
    inject_paths,
    persist_config,
)


TINY_MODEL_SRC = textwrap.dedent("""
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

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)


    class TinyDataModule(L.LightningDataModule):
        def __init__(self, batch_size: int = 4):
            super().__init__()
            self.batch_size = batch_size

        def setup(self, stage=None):
            torch.manual_seed(0)
            x = torch.randn(16, 4)
            y = torch.randint(0, 2, (16,))
            self.train_ds = TensorDataset(x[:12], y[:12])
            self.val_ds = TensorDataset(x[12:], y[12:])

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size)
""")


# Minimal defaults — drops anything that needs path injection so the test
# isolates the Lightning-seam questions rather than depending on the shape
# of src/mlops/defaults.yaml (which is iterating).
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
    data:
      class_path: tiny.TinyDataModule
      init_args:
        batch_size: 4
""")


@pytest.fixture
def _cleanup_sys_modules():
    """Ensure leftover 'tiny' module doesn't pollute later tests."""
    yield
    sys.modules.pop("tiny", None)


def test_smoke_fit_end_to_end(tmp_path, _cleanup_sys_modules, monkeypatch):
    # Mute Lightning's "num_workers" suggestion noise in CI.
    monkeypatch.setenv("PYTHONWARNINGS", "ignore::UserWarning")

    model_file = tmp_path / "tiny.py"
    model_file.write_text(TINY_MODEL_SRC)
    model_yaml = tmp_path / "tiny.yaml"
    model_yaml.write_text(MODEL_YAML)
    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text(DEFAULTS_YAML)

    # Resolve the model class. This populates sys.modules["tiny"] so the
    # `tiny.TinyDataModule` class_path in MODEL_YAML resolves.
    model_class = discover_model_class(model_file)
    assert model_class.__name__ == "TinyModel"

    # Build CLI. We pass defaults via the standard cascade by monkey-patching
    # _default_cascade_paths. Keeps the test independent of the real shipped
    # yamls, which are still iterating.
    import mlops.main as main_mod
    monkeypatch.setattr(
        main_mod, "_default_cascade_paths",
        lambda mf: [defaults_path, mf.with_suffix(".yaml")],
    )

    ft4_args = Ft4Args(
        subcommand="train",
        model_file=model_file,
        lightning_args=[],
    )
    cli = build_lightning_cli(ft4_args, model_class)

    # Set up the hset dir, create a session dir, and inject paths.
    hset_dir = tmp_path / "hset"
    (hset_dir / "checkpoints").mkdir(parents=True)
    session_dir = create_session_dir(hset_dir)
    inject_paths(cli.trainer, hset_dir, session_dir)

    # Persist config.
    persist_config(cli.parser, cli.config, hset_dir)

    # Fit.
    cli.trainer.fit(cli.model, cli.datamodule)

    # Rebuild the cumulative metrics file at hset level.
    rebuild_cumulative_metrics(hset_dir)

    # Artifacts.
    assert (hset_dir / "config.yaml").exists(), (
        "persist_config did not write config.yaml"
    )
    assert (hset_dir / "checkpoints" / "last.ckpt").exists(), (
        "ModelCheckpoint.dirpath mutation failed — last.ckpt not in <hset>/checkpoints/"
    )
    assert (session_dir / "metrics.csv").exists(), (
        "CSVLogger._save_dir mutation failed — metrics.csv not in <session>/"
    )
    assert (hset_dir / "metrics.csv").exists(), (
        "rebuild_cumulative_metrics did not write <hset>/metrics.csv"
    )
