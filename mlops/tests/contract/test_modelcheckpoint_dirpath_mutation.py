"""Contract: ModelCheckpoint reads `dirpath` lazily, allowing
post-instantiation mutation.

ft4 instantiates the checkpoint callback via the YAML cascade with
`dirpath=""` (placeholder), then mutates `ckpt_cb.dirpath` to the resolved
hset dir before calling `trainer.fit()`. This test pins that contract.

If this fails, ft4's `inject_paths` strategy must change. The likely fallback
is two-pass instantiation: instantiate the model alone, route to a hset dir,
then construct the trainer with the correct dirpath baked in from the start.
"""
import tempfile
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from conftest import TinyModel, TinyDataModule, silent_trainer_kwargs


def test_modelcheckpoint_dirpath_mutation():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "hset" / "checkpoints"

        # Cascade-default placeholder.
        ckpt_cb = ModelCheckpoint(
            dirpath="",
            save_top_k=1,
            monitor="train_loss",
            mode="min",
            save_last=True,
        )
        # Mutate before fit (ft4's inject_paths step).
        ckpt_cb.dirpath = str(target)

        trainer = L.Trainer(
            max_epochs=1,
            callbacks=[ckpt_cb],
            logger=False,
            **silent_trainer_kwargs(),
        )
        trainer.fit(TinyModel(), datamodule=TinyDataModule())

        assert target.exists(), (
            f"Expected dirpath mutation to take effect; "
            f"checkpoint dir {target} does not exist."
        )
        ckpts = list(target.glob("*.ckpt"))
        assert len(ckpts) >= 1, (
            f"Expected at least one .ckpt in {target}; "
            f"got {[p.name for p in target.iterdir()]}"
        )
        last_ckpt = target / "last.ckpt"
        assert last_ckpt.exists(), (
            f"Expected last.ckpt in {target}; "
            f"got {[p.name for p in target.iterdir()]}"
        )
