"""Integration regression tests for May 14 stress-test findings.

Covers:
  - BUG-2: phase-1 baseline failure must not kill the run.
  - BUG-5: a run that would do no work (already at max_epochs) must short-
    circuit before creating a session.

Both tests use the same TinyModel/TinyDataModule recipe as the smoke test.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlops.run import run
from mlops.hset_state import Ft4Args


# ── Test fixtures (parallel to test_run_smart_flow.py) ──────────────────────

# Model that defines generate_samples — but it raises. Exercises BUG-2 fix.
TINY_MODEL_WITH_FAILING_GENERATE_SRC = textwrap.dedent("""
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

        def generate_samples(self, prompts=None):
            # Simulates a real-world failure: tokenizer with surrogateescape
            # producing surrogate code points that pathlib can't UTF-8 encode.
            raise UnicodeEncodeError(
                "utf-8", "\\udcaf", 0, 1, "surrogates not allowed",
            )


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
def env_with_failing_generate(tmp_path, monkeypatch):
    """Workspace where the model's generate_samples raises."""
    model_file = tmp_path / "tiny.py"
    model_file.write_text(TINY_MODEL_WITH_FAILING_GENERATE_SRC)
    (tmp_path / "tiny.yaml").write_text(MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])

    sys.modules.pop("tiny", None)
    yield SimpleNamespace(
        model_file=model_file,
        runs_root=tmp_path / "runs",
    )
    sys.modules.pop("tiny", None)


def _args(env, **kw) -> Ft4Args:
    return Ft4Args(
        subcommand=kw.pop("subcommand", "train"),
        model_file=env.model_file,
        model_class=kw.pop("model_class", None),
        hset=kw.pop("hset", None),
        version=kw.pop("version", None),
        ckpt=kw.pop("ckpt", "last"),
        new_version=kw.pop("new_version", False),
        vdesc=kw.pop("vdesc", None),
        hdesc=kw.pop("hdesc", None),
        prompt=kw.pop("prompt", None),
        lightning_args=kw.pop("lightning_args", []),
    )


# ── BUG-2: phase-1 baseline failure does not kill the run ────────────────

class TestPhase1BaselineNonFatal:
    """If generate_samples raises during phase-1 baseline, the run continues:
    training happens, the session record is written, and the hset has its
    full audit trail. The phase-1 baseline is a UX nicety, not a correctness
    primitive — its failure must not propagate.

    Pre-fix, run.py would call write_sample_file directly with no exception
    handling, leaving the hset half-built with config.yaml + hset.json but
    no sessions.jsonl. See bugs.md BUG-2.
    """

    def test_run_completes_when_generate_samples_raises(
        self, env_with_failing_generate
    ):
        rc = run(_args(env_with_failing_generate),
                 runs_root=env_with_failing_generate.runs_root)
        assert rc == 0

    def test_session_record_written_when_baseline_fails(
        self, env_with_failing_generate
    ):
        run(_args(env_with_failing_generate),
            runs_root=env_with_failing_generate.runs_root)

        hset_dir = (
            env_with_failing_generate.runs_root / "tiny" / "v001" / "h001"
        )
        sessions_jsonl = hset_dir / "sessions.jsonl"
        assert sessions_jsonl.exists()
        records = [
            json.loads(l) for l in sessions_jsonl.read_text().splitlines()
        ]
        assert len(records) == 1
        # Training itself ran cleanly; baseline failure didn't poison it.
        assert records[0]["status"] == "completed"

    def test_baseline_failure_prints_warning(
        self, env_with_failing_generate, capsys
    ):
        """Student needs to know that samples won't be available for this
        hset; print the cause to stderr."""
        run(_args(env_with_failing_generate),
            runs_root=env_with_failing_generate.runs_root)

        captured = capsys.readouterr()
        # Look for some mention of baseline/samples failing.
        err = captured.err.lower()
        assert "baseline" in err or "sample" in err or "warning" in err

    def test_no_baseline_sample_file_on_failure(
        self, env_with_failing_generate
    ):
        """If write_sample_file raised, we don't want a half-written file."""
        run(_args(env_with_failing_generate),
            runs_root=env_with_failing_generate.runs_root)

        hset_dir = (
            env_with_failing_generate.runs_root / "tiny" / "v001" / "h001"
        )
        baseline = hset_dir / "samples" / "step_000000.md"
        assert not baseline.exists()


# ── BUG-5: no-op runs short-circuit before creating a session ──────────────

# A normal model (no generate_samples) for BUG-5 tests — reuse the recipe
# from test_run_smart_flow but with that method absent.
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


@pytest.fixture
def env_normal(tmp_path, monkeypatch):
    """Workspace with a normal TinyModel (no generate_samples)."""
    model_file = tmp_path / "tiny.py"
    model_file.write_text(TINY_MODEL_SRC)
    (tmp_path / "tiny.yaml").write_text(MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])

    sys.modules.pop("tiny", None)
    yield SimpleNamespace(
        model_file=model_file,
        runs_root=tmp_path / "runs",
    )
    sys.modules.pop("tiny", None)


class TestNoOpShortCircuit:
    """A second invocation with no work to do (already reached max_epochs)
    must detect that and exit gracefully — not create an empty session dir.

    Pre-fix: Lightning's fit() exits immediately when current_epoch >=
    max_epochs, but session_dir was already created, hparams.yaml was already
    written by the logger, and a metrics.csv-less stub remained. The tree
    accumulated dozens of these. See bugs.md BUG-5.
    """

    def test_second_run_at_same_max_epochs_creates_no_session(self, env_normal):
        """First run trains to max_epochs=1. Second run with same settings
        sees nothing to do — should create no new session."""
        # First run does real training.
        rc1 = run(_args(env_normal), runs_root=env_normal.runs_root)
        assert rc1 == 0

        hset_dir = env_normal.runs_root / "tiny" / "v001" / "h001"
        sessions_before = list((hset_dir / "sessions").iterdir())
        assert len(sessions_before) == 1, "First run should have created s001"

        # Second run: same args, should detect 'already done' and exit.
        rc2 = run(_args(env_normal), runs_root=env_normal.runs_root)
        assert rc2 == 0

        sessions_after = list((hset_dir / "sessions").iterdir())
        # No new session_dir created.
        assert len(sessions_after) == 1

    def test_no_op_run_prints_helpful_message(self, env_normal, capsys):
        """The student should see a clear 'nothing to do' message, not
        silent success."""
        run(_args(env_normal), runs_root=env_normal.runs_root)
        capsys.readouterr()  # discard first run's output

        run(_args(env_normal), runs_root=env_normal.runs_root)
        captured = capsys.readouterr()
        err = captured.err.lower()
        # Phrasing flexible; concept fixed.
        assert (
            "nothing to do" in err
            or "already" in err
            or "complete" in err
            or "no work" in err
        )

    def test_no_op_run_does_not_append_to_sessions_jsonl(self, env_normal):
        """A no-op run did no work; the audit trail should not record it
        (the previous completed session already represents the current state)."""
        run(_args(env_normal), runs_root=env_normal.runs_root)

        hset_dir = env_normal.runs_root / "tiny" / "v001" / "h001"
        sessions_jsonl = hset_dir / "sessions.jsonl"
        records_before = len(sessions_jsonl.read_text().splitlines())

        run(_args(env_normal), runs_root=env_normal.runs_root)

        records_after = len(sessions_jsonl.read_text().splitlines())
        assert records_after == records_before

    def test_bumped_max_epochs_still_trains(self, env_normal):
        """Bumping max_epochs above current epoch is NOT a no-op; training
        should proceed for the additional epoch."""
        run(_args(env_normal), runs_root=env_normal.runs_root)

        hset_dir = env_normal.runs_root / "tiny" / "v001" / "h001"

        # Second run: bump max_epochs from 1 to 2 via CLI override.
        rc = run(
            _args(env_normal, lightning_args=["--trainer.max_epochs=2"]),
            runs_root=env_normal.runs_root,
        )
        assert rc == 0

        sessions = list((hset_dir / "sessions").iterdir())
        # New session created because there was real work to do.
        assert len(sessions) == 2

    def test_previous_errored_session_does_not_short_circuit(
        self, tmp_path, env_normal
    ):
        """If the last session ended with an error, a re-run should NOT
        short-circuit — the previous failure means there's still work to do."""
        # Manually inject a previously-errored session into a fresh hset.
        hset_dir = env_normal.runs_root / "tiny" / "v001" / "h001"

        # First, do one real run to create the directory structure.
        run(_args(env_normal), runs_root=env_normal.runs_root)

        # Now overwrite the session record to look like an error.
        sessions_jsonl = hset_dir / "sessions.jsonl"
        records = [
            json.loads(l) for l in sessions_jsonl.read_text().splitlines()
        ]
        records[-1]["status"] = "error"
        records[-1]["epochs"] = 0  # the error prevented progress
        records[-1]["steps"] = 0
        records[-1]["error"] = "TestError: simulated"
        sessions_jsonl.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        # Re-run should NOT short-circuit; previous attempt failed.
        rc = run(_args(env_normal), runs_root=env_normal.runs_root)
        assert rc == 0

        sessions = list((hset_dir / "sessions").iterdir())
        # A new session attempt was made.
        assert len(sessions) >= 2
