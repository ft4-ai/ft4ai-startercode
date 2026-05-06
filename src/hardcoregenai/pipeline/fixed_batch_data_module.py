"""
A tiny Lightning DataModule for tests that yields exactly the batches you pass in.
It mirrors StoriesDataModule's batch format: dicts with a "tokens" LongTensor of shape (B, L+1).
"""
from typing import Optional, Dict
import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class FixedBatchDataModule(L.LightningDataModule):
    """Minimal DataModule for unit tests.

    Args:
        train_tokens: LongTensor[B, L+1] for training.
        val_tokens:   Optional LongTensor[B, L+1] for validation.
        test_tokens:  Optional LongTensor[B, L+1] for test.
    """
    def __init__(
        self,
        train_tokens: Tensor,
        val_tokens: Optional[Tensor] = None,
        test_tokens: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self._train_ds = _SingleBatchDataset(train_tokens)
        self._val_ds = _SingleBatchDataset(val_tokens) if val_tokens is not None else None
        self._test_ds = _SingleBatchDataset(test_tokens) if test_tokens is not None else None
        self._num_workers = 0
        self._pin_memory = False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=None,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self):
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=None,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        if self._test_ds is None:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=None,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

class _SingleBatchDataset(Dataset):
    """Dataset that returns a single pre-baked batch dict: {"tokens": LongTensor[B, L+1]}"""
    def __init__(self, tokens: Tensor):
        assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long and tokens.ndim == 2, (
            "tokens must be LongTensor of shape (B, L+1)"
        )
        self._batch = {"tokens": tokens}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Ignore idx; always return the single batch
        return self._batch
