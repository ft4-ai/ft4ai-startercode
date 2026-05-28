"""Shared fixtures and helpers for ft4 contract tests.

Pytest auto-discovers this file; the test modules import directly from it
(e.g. `from conftest import TinyModel`), which works under pytest's standard
rootdir mode since the directory is added to sys.path during test collection.

This module is deliberately Lightning-only — no ft4 imports — so the tests
remain valid in isolation, before any ft4 code exists.
"""
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(L.LightningModule):
    """Minimal LightningModule with one tunable hparam.

    `hidden` is exposed so test_lightning_cli_run_false can verify
    --model.hidden=N flows through the cascade correctly.
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TinyDataModule(L.LightningDataModule):
    """Deterministic 16-sample synthetic dataset."""

    def __init__(self, batch_size: int = 4, n_samples: int = 16):
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples

    def setup(self, stage=None):
        gen = torch.Generator().manual_seed(42)
        self.x = torch.randn(self.n_samples, 4, generator=gen)
        self.y = torch.randn(self.n_samples, 2, generator=gen)

    def _loader(self):
        return DataLoader(TensorDataset(self.x, self.y), batch_size=self.batch_size)

    def train_dataloader(self):
        return self._loader()

    def val_dataloader(self):
        return self._loader()


def silent_trainer_kwargs() -> dict:
    """Trainer kwargs that quiet down progress bar and model summary
    for contract tests. Does not touch logger or checkpointing — each
    test configures those itself."""
    return dict(
        enable_progress_bar=False,
        enable_model_summary=False,
    )
