"""Integration tests for the smart-flow `run()` orchestration.

Heavy: each test invokes a real LightningCLI + trainer.fit on a tiny
model. Kept under 10s total. Builds directly on the smoke test's
TinyModel/TinyDataModule recipe.
"""
import json
import sys
import textwrap
from pathlib import Path

import pytest

from mlops.run import run
from mlops.hset_state import Ft4Args


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
def env(tmp_path, monkeypatch):
    """Set up a complete ft4 workspace in tmp_path: model file, model.yaml,
    custom defaults, and a runs/ root."""
    model_file = tmp_path / "tiny.py"
    model_file.write_text(TINY_MODEL_SRC)
    (tmp_path / "tiny.yaml").write_text(MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])

    # Drop any cached 'tiny' module from prior tests in the same process.
    sys.modules.pop("tiny", None)
    yield SimpleEnv(
        model_file=model_file,
        runs_root=tmp_path / "runs",
        tmp_path=tmp_path,
    )
    sys.modules.pop("tiny", None)


class SimpleEnv:
    def __init__(self, model_file: Path, runs_root: Path, tmp_path: Path):
        self.model_file = model_file
        self.runs_root = runs_root
        self.tmp_path = tmp_path


def _args(env: SimpleEnv, **kw) -> Ft4Args:
    return Ft4Args(
        subcommand=kw.pop("subcommand", "train"),
        model_file=env.model_file,
        lightning_args=kw.pop("lightning_args", []),
        **kw,
    )


# ── Single invocation ─────────────────────────────────────────────────────

def test_run_creates_full_hset_layout(env):
    rc = run(_args(env), runs_root=env.runs_root)
    assert rc == 0

    hset_dir = env.runs_root / "tiny" / "v001" / "h001"
    assert hset_dir.is_dir()

    # Version-level artifacts.
    version_dir = env.runs_root / "tiny" / "v001"
    assert (version_dir / "version.json").is_file()
    assert (version_dir / "model.py.snapshot").is_file()

    # Hset-level artifacts.
    assert (hset_dir / "hset.json").is_file()
    assert (hset_dir / "config.yaml").is_file()
    assert (hset_dir / "sessions.jsonl").is_file()

    # Per-session artifacts.
    assert (hset_dir / "sessions" / "s001" / "metrics.csv").is_file()
    assert (hset_dir / "sessions" / "s001" / "runconfig.yaml").is_file()

    # Cumulative.
    assert (hset_dir / "metrics.csv").is_file()

    # Checkpoint at hset level.
    assert (hset_dir / "checkpoints" / "last.ckpt").is_file()


def test_run_writes_session_record_with_expected_fields(env):
    run(_args(env), runs_root=env.runs_root)

    sessions_jsonl = env.runs_root / "tiny" / "v001" / "h001" / "sessions.jsonl"
    records = [json.loads(line) for line in sessions_jsonl.read_text().splitlines()]
    assert len(records) == 1
    rec = records[0]
    assert rec["session_name"] == "s001"
    assert rec["session_index"] == 1
    assert rec["epochs"] >= 0
    assert rec["steps"] > 0
    assert "started_at" in rec and "ended_at" in rec
    # Metric labels carry the actual logged key (e.g. `train_ce_epoch`),
    # so the briefing renders honestly.
    assert isinstance(rec["train_metric_key"], str)
    assert rec["train_metric_value"] is not None
    assert isinstance(rec["val_metric_key"], str)
    assert rec["val_metric_value"] is not None
    assert "all_metrics" in rec
    # Outside a git repo — both None.
    assert rec["git_commit"] is None
    assert rec["git_dirty"] is None


# ── Multi-session accumulation ────────────────────────────────────────────

def test_two_runs_same_config_reuse_hset_and_accumulate(env):
    # First session trains to epoch 1, saves last.ckpt.
    run(_args(env, lightning_args=["--trainer.max_epochs=1"]), runs_root=env.runs_root)
    # Second session resumes from last.ckpt. Must bump max_epochs or
    # Lightning will see "already at max_epochs, done" and emit nothing —
    # which is correct semantics but useless for testing accumulation.
    run(_args(env, lightning_args=["--trainer.max_epochs=2"]), runs_root=env.runs_root)

    hset_dir = env.runs_root / "tiny" / "v001" / "h001"

    # Two session directories.
    sessions_root = hset_dir / "sessions"
    session_dirs = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    assert session_dirs == ["s001", "s002"]

    # Two session records.
    records = [
        json.loads(line)
        for line in (hset_dir / "sessions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["session_name"] == "s001"
    assert records[1]["session_name"] == "s002"

    # Cumulative metrics contains both sessions.
    cumulative = (hset_dir / "metrics.csv").read_text().splitlines()
    sessions_in_cumulative = {
        line.split(",")[0]  # 'session' is the leading column
        for line in cumulative[1:]
    }
    assert sessions_in_cumulative == {"1", "2"}


# ── --compile ─────────────────────────────────────────────────────────────

def test_compile_flag_invokes_torch_compile(env, monkeypatch):
    """With `--compile`, run() should call torch.compile on cli.model before
    fit(). We monkeypatch torch.compile to a no-op spy so the test stays fast.
    """
    import torch
    called = {}

    def spy_compile(model, *a, **kw):
        called["yes"] = True
        return model  # return unwrapped; we don't actually want to compile

    monkeypatch.setattr(torch, "compile", spy_compile)
    rc = run(_args(env, compile=True), runs_root=env.runs_root)
    assert rc == 0
    assert called.get("yes") is True


def test_no_compile_flag_does_not_invoke_torch_compile(env, monkeypatch):
    import torch
    called = {}

    def spy_compile(model, *a, **kw):
        called["yes"] = True
        return model

    monkeypatch.setattr(torch, "compile", spy_compile)
    rc = run(_args(env), runs_root=env.runs_root)
    assert rc == 0
    assert "yes" not in called


# ── --hset auto-load (integration) ────────────────────────────────────────

def test_hset_autoload_resumes_after_model_yaml_edit(env):
    """The student's most common pain point: train h001, later edit
    model.yaml (e.g. bump hidden dim), then try --hset h001 → currently
    that errored with a hash mismatch. With auto-load it just works."""
    # Session 1: create h001 with the original model.yaml.
    rc = run(_args(env), runs_root=env.runs_root)
    assert rc == 0
    hset_dir = env.runs_root / "tiny" / "v001" / "h001"
    assert (hset_dir / "config.yaml").is_file()
    sessions_jsonl_before = (hset_dir / "sessions.jsonl").read_text()

    # Now the student edits model.yaml (bumps hidden).
    (env.tmp_path / "tiny.yaml").write_text(textwrap.dedent("""
        model:
          hidden: 32
        data:
          class_path: tiny.TinyDataModule
          init_args:
            batch_size: 4
    """))

    # Resume h001. Without auto-load this raises "hparam hash mismatch".
    # With auto-load it picks up h001's saved config.yaml and resumes cleanly.
    rc = run(
        _args(env, hset="h001", lightning_args=["--trainer.max_epochs=2"]),
        runs_root=env.runs_root,
    )
    assert rc == 0
    # A new session was appended to the *same* hset (no fork).
    sessions_after = (hset_dir / "sessions.jsonl").read_text()
    assert sessions_after != sessions_jsonl_before
    assert sessions_after.count("\n") == 2  # two session records


