"""Contract: CSVLogger appends to existing metrics.csv across separate
Python processes.

ft4 hsets accumulate metrics across multiple sessions, each session being a
separate `ft4 run` invocation in a new Python process. metrics.csv must
accumulate rows across these sessions. If Lightning's CSVLogger truncates
instead, ft4's `TimestampedCSVLogger` must implement explicit append behavior
in its `__init__`.

This test runs two child processes in sequence, each pointing at the same
save_dir, and asserts that metrics.csv grows.
"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
import pytest

CHILD_SCRIPT = textwrap.dedent('''
    """Child: run one fit() and exit."""
    import sys
    import torch
    import torch.nn as nn
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset

    class M(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.mse_loss(self.fc(x), y)
            self.log("train_loss", loss, on_step=True, on_epoch=False)
            return loss
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    save_dir = sys.argv[1]
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(8, 4, generator=gen)
    y = torch.randn(8, 2, generator=gen)
    dl = DataLoader(TensorDataset(x, y), batch_size=4)

    logger = CSVLogger(save_dir=save_dir, name="", version="")
    trainer = L.Trainer(
        max_epochs=1, logger=logger,
        enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=False, log_every_n_steps=1,
    )
    trainer.fit(M(), dl)
''')

@pytest.mark.xfail(
    reason="Lightning's stock CSVLogger truncates on second session; "
           "TimestampedCSVLogger preloads to fake append. Re-test with the "
           "subclass once step 7 is done.",
    strict=True,
)
def test_csvlogger_cross_process_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "child.py"
        script_path.write_text(CHILD_SCRIPT)
        save_dir = tmpdir_path / "hset"
        save_dir.mkdir()

        def run_child():
            result = subprocess.run(
                [sys.executable, str(script_path), str(save_dir)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"child failed (rc={result.returncode})\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

        # Session 1.
        run_child()
        metrics = save_dir / "metrics.csv"
        assert metrics.exists(), "metrics.csv missing after first session"
        rows_after_1 = len(metrics.read_text().splitlines())
        assert rows_after_1 >= 2, (
            f"first session wrote too few rows: {rows_after_1}"
        )

        # Session 2.
        run_child()
        rows_after_2 = len(metrics.read_text().splitlines())

        assert rows_after_2 > rows_after_1, (
            f"Expected metrics.csv to accumulate across sessions; "
            f"got {rows_after_1} → {rows_after_2} rows. "
            f"TimestampedCSVLogger must implement explicit append behavior. "
            f"See README.md for the fallback design."
        )
