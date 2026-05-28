"""Unit tests for SampleGenerationCallback.

Uses lightweight SimpleNamespace fakes for trainer and pl_module. The full
Lightning-loop integration is exercised by
tests/integration/test_run_smart_flow_generative.py.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mlops.sample_callback import SampleGenerationCallback
from mlops.sampling import write_sample_file


def _trainer(current_epoch: int = 0, global_step: int = 0,
             sanity_checking: bool = False,
             loggers: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        current_epoch=current_epoch,
        global_step=global_step,
        sanity_checking=sanity_checking,
        loggers=loggers if loggers is not None else [],
    )


def _generative_model() -> SimpleNamespace:
    """A fake LightningModule (no specific methods needed; LangGen will be mocked)."""
    return SimpleNamespace()


def _classifier_model() -> SimpleNamespace:
    """A fake LightningModule with NO generate_samples method."""
    return SimpleNamespace()  # no attributes


# ── write_sample_file ──────────────────────────────────────────────────────

def test_write_sample_file_creates_dir(tmp_path):
    p = write_sample_file(tmp_path, step=42, text="hello")
    assert p == tmp_path / "samples" / "step_000042.md"
    assert p.read_text() == "hello"
    assert (tmp_path / "samples").is_dir()


def test_write_sample_file_zero_padding(tmp_path):
    write_sample_file(tmp_path, step=0, text="a")
    write_sample_file(tmp_path, step=1, text="b")
    write_sample_file(tmp_path, step=100, text="c")
    names = sorted(p.name for p in (tmp_path / "samples").iterdir())
    assert names == ["step_000000.md", "step_000001.md", "step_000100.md"]


def test_write_sample_file_overwrites(tmp_path):
    write_sample_file(tmp_path, step=5, text="first")
    write_sample_file(tmp_path, step=5, text="second")
    assert (tmp_path / "samples" / "step_000005.md").read_text() == "second"


def test_write_sample_file_six_digit_overflow(tmp_path):
    """1_000_000+ steps render as wider strings, still sort lexicographically
    after the 6-digit names."""
    p = write_sample_file(tmp_path, step=1_000_000, text="big")
    assert p.name == "step_1000000.md"


# ── Validation-epoch trigger ───────────────────────────────────────────────

def test_epoch_trigger_fires_every_validation_by_default(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=1, step_interval=0)
    cb.hset_dir = tmp_path
    model = _generative_model()

    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        cb.on_validation_epoch_end(_trainer(global_step=10), model)
        cb.on_validation_epoch_end(_trainer(global_step=20), model)

    names = sorted(p.name for p in (tmp_path / "samples").iterdir())
    assert names == ["step_000010.md", "step_000020.md"]


def test_epoch_trigger_respects_interval(tmp_path):
    """With epoch_interval=2, fires every 2nd validation: at calls 2, 4, ..."""
    cb = SampleGenerationCallback(epoch_interval=2, step_interval=0)
    cb.hset_dir = tmp_path
    model = _generative_model()

    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        for step in [10, 20, 30, 40]:
            cb.on_validation_epoch_end(_trainer(global_step=step), model)
    names = sorted(p.name for p in (tmp_path / "samples").iterdir())
    assert names == ["step_000020.md", "step_000040.md"]


def test_epoch_trigger_zero_disables(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=0, step_interval=0)
    cb.hset_dir = tmp_path
    cb.on_validation_epoch_end(_trainer(global_step=10), _generative_model())
    assert not (tmp_path / "samples").exists()


def test_sanity_check_validation_is_skipped(tmp_path):
    """Lightning runs ~2 batches of validation before training starts as a
    sanity check; samples from a freshly-initialized model are noise."""
    cb = SampleGenerationCallback(epoch_interval=1, step_interval=0)
    cb.hset_dir = tmp_path
    model = _generative_model()

    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        cb.on_validation_epoch_end(_trainer(global_step=0, sanity_checking=True), model)
        assert not (tmp_path / "samples").exists()

        # And the val_count must not have been bumped — first real validation
        # should still be call #1, which fires with interval=1.
        cb.on_validation_epoch_end(_trainer(global_step=5), model)
    assert (tmp_path / "samples" / "step_000005.md").is_file()


# ── Step trigger ───────────────────────────────────────────────────────────

def test_step_trigger_fires_at_interval(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=0, step_interval=100)
    cb.hset_dir = tmp_path
    model = _generative_model()

    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        for step in [50, 100, 150, 200]:
            cb.on_train_batch_end(_trainer(global_step=step), model, None, None, 0)

    names = sorted(p.name for p in (tmp_path / "samples").iterdir())
    assert names == ["step_000100.md", "step_000200.md"]


def test_step_trigger_zero_disables(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=0, step_interval=0)
    cb.hset_dir = tmp_path
    cb.on_train_batch_end(_trainer(global_step=100), _generative_model(), None, None, 0)
    assert not (tmp_path / "samples").exists()


def test_step_trigger_skips_step_zero(tmp_path):
    """Step 0 is reserved for the phase-1 baseline written by run.py."""
    cb = SampleGenerationCallback(epoch_interval=0, step_interval=1)
    cb.hset_dir = tmp_path
    cb.on_train_batch_end(_trainer(global_step=0), _generative_model(), None, None, 0)
    assert not (tmp_path / "samples").exists()


# ── Both triggers active ──────────────────────────────────────────────────

def test_both_triggers_fire_independently(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=1, step_interval=50)
    cb.hset_dir = tmp_path
    model = _generative_model()

    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        cb.on_train_batch_end(_trainer(global_step=50), model, None, None, 0)   # step trigger
        cb.on_train_batch_end(_trainer(global_step=100), model, None, None, 0)  # step trigger
        cb.on_validation_epoch_end(_trainer(global_step=125), model)  # epoch trigger

    names = sorted(p.name for p in (tmp_path / "samples").iterdir())
    assert names == ["step_000050.md", "step_000100.md", "step_000125.md"]


# ── Duck-typing on the model ──────────────────────────────────────────────

def test_silently_skips_model_without_generate_samples(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=1)
    cb.hset_dir = tmp_path
    cb.on_validation_epoch_end(_trainer(global_step=10), _classifier_model())
    cb.on_train_batch_end(_trainer(global_step=100), _classifier_model(), None, None, 0)
    assert not (tmp_path / "samples").exists()


# ── Prompts passthrough ───────────────────────────────────────────────────

def test_prompts_passed_to_generate_samples(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=1)
    cb.hset_dir = tmp_path
    cb.prompts = ["hello", "world"]
    model = _generative_model()
    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        cb.on_validation_epoch_end(_trainer(global_step=10), model)
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        assert call_args.kwargs["prompts"] == ["hello", "world"]


def test_no_prompts_passes_none(tmp_path):
    cb = SampleGenerationCallback(epoch_interval=1)
    cb.hset_dir = tmp_path
    # prompts default = None
    model = _generative_model()
    with patch("mlops.sampling.generate_sample_markdown") as mock_gen:
        mock_gen.return_value = "# samples\n"
        cb.on_validation_epoch_end(_trainer(global_step=10), model)
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        assert call_args.kwargs["prompts"] is None


# ── Defensive guards ──────────────────────────────────────────────────────

def test_no_hset_dir_silently_skips(tmp_path):
    """If hset_dir was never injected (broken wiring), callback no-ops
    rather than crashing."""
    cb = SampleGenerationCallback(epoch_interval=1)
    # cb.hset_dir is None
    cb.on_validation_epoch_end(_trainer(global_step=10), _generative_model())
    # No crash. (Nothing to assert about filesystem; tmp_path is untouched.)
    assert list(tmp_path.iterdir()) == []
