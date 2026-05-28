"""Integration test: smart flow with a generative model.

Exercises:
  - Phase 1 baseline (step_000000.md written before training)
  - SampleGenerationCallback during training (one file per epoch)
  - Prompts plumbing through Ft4Args → callback → model
  - Skip-on-resume behavior for phase 1 baseline
"""
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mlops import sampling
from mlops.run import run
from mlops.hset_state import Ft4Args


# A generative twin of TinyModel. generate_samples returns markdown showing
# whatever prompts (or "default") it received, so tests can assert on prompt
# plumbing by reading the file contents back.
TINY_LM_SRC = textwrap.dedent("""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import lightning as L

    class TinyLM(L.LightningModule):
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
        - class_path: mlops.sample_callback.SampleGenerationCallback
          init_args:
            epoch_interval: 1
            step_interval: 0
""")


MODEL_YAML = textwrap.dedent("""
    data:
      class_path: tiny_lm.TinyDataModule
      init_args:
        batch_size: 4
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny_lm.py"
    model_file.write_text(TINY_LM_SRC)
    (tmp_path / "tiny_lm.yaml").write_text(MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])
    sys.modules.pop("tiny_lm", None)
    yield SimpleEnv(model_file=model_file, runs_root=tmp_path / "runs")
    sys.modules.pop("tiny_lm", None)


class SimpleEnv:
    def __init__(self, model_file: Path, runs_root: Path):
        self.model_file = model_file
        self.runs_root = runs_root


def _make_mock_langgen_generate():
    """Create a mock function that returns markdown with prompts."""
    def _mock(*args, **kwargs):
        prompts = kwargs.get("prompts")
        tag = "default" if prompts is None else ",".join(prompts)
        return f"# samples\n\nprompts={tag}\n"
    return _mock


def _args(env: SimpleEnv, **kw) -> Ft4Args:
    return Ft4Args(
        subcommand=kw.pop("subcommand", "train"),
        model_file=env.model_file,
        lightning_args=kw.pop("lightning_args", []),
        **kw,
    )


# ── Phase 1 baseline ──────────────────────────────────────────────────────

def test_phase_1_baseline_written_for_fresh_hset(env):
    with patch.object(sampling, "generate_sample_markdown", side_effect=_make_mock_langgen_generate()):
        run(_args(env), runs_root=env.runs_root)

    hset_dir = env.runs_root / "tiny_lm" / "v001" / "h001"
    baseline = hset_dir / "samples" / "step_000000.md"
    assert baseline.is_file()
    assert "prompts=default" in baseline.read_text()


def test_phase_1_baseline_uses_provided_prompts(env):
    with patch.object(sampling, "generate_sample_markdown", side_effect=_make_mock_langgen_generate()):
        run(_args(env, prompt=["alpha", "beta"]), runs_root=env.runs_root)

    hset_dir = env.runs_root / "tiny_lm" / "v001" / "h001"
    baseline = (hset_dir / "samples" / "step_000000.md").read_text()
    assert "prompts=alpha,beta" in baseline


def test_phase_1_skipped_on_resume(env):
    with patch.object(sampling, "generate_sample_markdown", side_effect=_make_mock_langgen_generate()):
        # Session 1: train for 1 epoch.
        run(_args(env, lightning_args=["--trainer.max_epochs=1"]), runs_root=env.runs_root)
        hset_dir = env.runs_root / "tiny_lm" / "v001" / "h001"
        baseline_path = hset_dir / "samples" / "step_000000.md"
        mtime_after_session_1 = baseline_path.stat().st_mtime_ns

        # Sleep then session 2: resumes from last.ckpt. Phase 1 baseline should
        # NOT fire (we'd otherwise overwrite step_000000.md).
        import time
        time.sleep(0.01)
        run(_args(env, lightning_args=["--trainer.max_epochs=2"]), runs_root=env.runs_root)

        assert baseline_path.stat().st_mtime_ns == mtime_after_session_1, (
            "phase 1 baseline ran on resumed session; should be skipped"
        )


# ── Sample callback during training ───────────────────────────────────────

def test_callback_writes_sample_per_epoch(env):
    with patch.object(sampling, "generate_sample_markdown", side_effect=_make_mock_langgen_generate()):
        run(
            _args(env, lightning_args=["--trainer.max_epochs=2"]),
            runs_root=env.runs_root,
        )

    samples_dir = env.runs_root / "tiny_lm" / "v001" / "h001" / "samples"
    names = sorted(p.name for p in samples_dir.iterdir())
    # Phase 1 baseline (step 0) + one per epoch (steps 3 and 6 here, since
    # the TinyDataModule has 3 train batches per epoch).
    assert "step_000000.md" in names  # phase 1 baseline
    # At least one non-baseline file from the callback.
    non_baseline = [n for n in names if n != "step_000000.md"]
    assert len(non_baseline) >= 1, f"callback did not write any samples: {names}"


def test_callback_uses_prompts_from_ft4_args(env):
    with patch.object(sampling, "generate_sample_markdown", side_effect=_make_mock_langgen_generate()):
        run(
            _args(env, prompt=["gamma"], lightning_args=["--trainer.max_epochs=1"]),
            runs_root=env.runs_root,
        )

    samples_dir = env.runs_root / "tiny_lm" / "v001" / "h001" / "samples"
    # Every sample file (baseline + callback) should reflect the prompt.
    for path in samples_dir.iterdir():
        assert "prompts=gamma" in path.read_text(), (
            f"{path.name} did not get the gamma prompt"
        )


# ── Non-generative model: callback no-ops ─────────────────────────────────

NON_GEN_MODEL_SRC = textwrap.dedent("""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import lightning as L

    class TinyClassifier(L.LightningModule):
        def __init__(self, hidden: int = 8):
            super().__init__()
            self.save_hyperparameters()
            self.fc = nn.Linear(4, hidden)
            self.out = nn.Linear(hidden, 2)

        def forward(self, x): return self.out(self.fc(x))

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
        # NO generate_samples method.


    class TinyDataModule(L.LightningDataModule):
        def __init__(self, batch_size: int = 4):
            super().__init__()
            self.batch_size = batch_size

        def setup(self, stage=None):
            torch.manual_seed(0)
            x = torch.randn(16, 4); y = torch.randint(0, 2, (16,))
            self.train_ds = TensorDataset(x[:12], y[:12])
            self.val_ds = TensorDataset(x[12:], y[12:])

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size)
""")


NON_GEN_MODEL_YAML = textwrap.dedent("""
    data:
      class_path: tiny_clf.TinyDataModule
      init_args:
        batch_size: 4
""")


@pytest.fixture
def clf_env(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny_clf.py"
    model_file.write_text(NON_GEN_MODEL_SRC)
    (tmp_path / "tiny_clf.yaml").write_text(NON_GEN_MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(DEFAULTS_YAML)

    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])
    sys.modules.pop("tiny_clf", None)
    yield SimpleEnv(model_file=model_file, runs_root=tmp_path / "runs")
    sys.modules.pop("tiny_clf", None)


def test_non_generative_model_skips_samples_entirely(clf_env):
    """Classifier with no generate_samples: no samples/ dir is created,
    no errors, training completes normally."""
    rc = run(_args(clf_env), runs_root=clf_env.runs_root)
    assert rc == 0

    hset_dir = clf_env.runs_root / "tiny_clf" / "v001" / "h001"
    # Hset directory exists, training completed.
    assert (hset_dir / "checkpoints" / "last.ckpt").is_file()
    # But samples/ was never created — nothing wrote to it.
    assert not (hset_dir / "samples").exists()
