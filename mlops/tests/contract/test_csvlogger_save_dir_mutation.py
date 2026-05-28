"""Contract: CSVLogger reads `_save_dir` lazily, allowing post-instantiation
mutation before the first `log_metrics` call.

ft4 instantiates the logger via the YAML cascade with `save_dir=""` and
`name=""`, `version=""`, then mutates `logger._save_dir` to the resolved
hset dir before any training begins. With name and version both empty,
`log_dir` resolves to `_save_dir/` so `metrics.csv` lands directly in the
hset directory.

If this fails, the alternative is the same two-pass approach as
test_modelcheckpoint_dirpath_mutation, or subclassing CSVLogger with
sentinel resolution at log time.
"""
import tempfile
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from conftest import TinyModel, TinyDataModule, silent_trainer_kwargs


def test_csvlogger_save_dir_mutation():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "hset"
        target.mkdir()

        # Cascade-default placeholder with name and version both empty so the
        # logger does NOT create version_N/ subdirectories.
        logger = CSVLogger(save_dir="", name="", version="")
        logger._save_dir = str(target)

        trainer = L.Trainer(
            max_epochs=1,
            logger=logger,
            enable_checkpointing=False,
            log_every_n_steps=1,
            **silent_trainer_kwargs(),
        )
        trainer.fit(TinyModel(), datamodule=TinyDataModule())

        metrics = target / "metrics.csv"
        assert metrics.exists(), (
            f"Expected metrics.csv directly in mutated save_dir {target}; "
            f"got: {[p.relative_to(target) for p in target.rglob('*')]}"
        )
        lines = metrics.read_text().splitlines()
        assert len(lines) >= 2, (
            f"metrics.csv too short (need header + 1 row); got {lines}"
        )
