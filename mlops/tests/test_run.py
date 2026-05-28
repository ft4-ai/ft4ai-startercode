"""Unit tests for mlops.run helpers.

The orchestration in `run()` itself is covered by
tests/integration/test_run_smart_flow.py — that path exercises real
Lightning + tmp filesystem and is too heavy for unit tests.
"""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlops.run import (
    _build_session_record,
    _flatten_callback_metrics,
    _pick_metric,
    _print_session_summary,
    _to_float,
    _TRAIN_METRIC_BASES,
    _VAL_METRIC_BASES,
)


# ── _to_float ─────────────────────────────────────────────────────────────

def test_to_float_python_int():
    assert _to_float(3) == 3.0


def test_to_float_python_float():
    assert _to_float(3.5) == 3.5


def test_to_float_tensor_scalar():
    assert _to_float(torch.tensor(1.25)) == pytest.approx(1.25)


def test_to_float_tensor_int_scalar():
    assert _to_float(torch.tensor(7)) == 7.0


def test_to_float_unconvertible_returns_none():
    # Unconvertible inputs return None so callers can skip them. Beats
    # returning repr(v), which would smuggle a string into a float-typed dict.
    assert _to_float(object()) is None


def test_flatten_callback_metrics_skips_unconvertible():
    cm = {"train_loss": torch.tensor(0.5), "weird": object()}
    out = _flatten_callback_metrics(cm)
    assert out == pytest.approx({"train_loss": 0.5})


# ── _pick_metric (suffix priority: _epoch > "" > _step) ───────────────────

def test_pick_metric_prefers_epoch():
    """Both _step and _epoch present → _epoch wins."""
    d = {"train_ce_step": 5.0, "train_ce_epoch": 1.0}
    assert _pick_metric(d, _TRAIN_METRIC_BASES) == ("train_ce_epoch", 1.0)


def test_pick_metric_unsuffixed_when_no_epoch():
    """No _epoch, no _step → bare key wins."""
    d = {"train_ce": 2.5}
    assert _pick_metric(d, _TRAIN_METRIC_BASES) == ("train_ce", 2.5)


def test_pick_metric_falls_back_to_step_when_thats_all():
    """Only _step present (e.g., on_step=True only) → _step wins."""
    d = {"train_ce_step": 3.5}
    assert _pick_metric(d, _TRAIN_METRIC_BASES) == ("train_ce_step", 3.5)


def test_pick_metric_first_base_name_wins():
    """If both train_ce_epoch and train_loss_epoch exist, ce wins."""
    d = {"train_ce_epoch": 1.0, "train_loss_epoch": 2.0}
    assert _pick_metric(d, _TRAIN_METRIC_BASES) == ("train_ce_epoch", 1.0)


def test_pick_metric_none_when_no_match():
    assert _pick_metric({"other": 1.0}, _TRAIN_METRIC_BASES) is None


def test_pick_metric_val():
    d = {"val_ce_epoch": 0.42}
    assert _pick_metric(d, _VAL_METRIC_BASES) == ("val_ce_epoch", 0.42)


# ── _flatten_callback_metrics ─────────────────────────────────────────────

def test_flatten_callback_metrics_tensors():
    cm = {"train_loss": torch.tensor(0.5), "val_loss": torch.tensor(0.7)}
    out = _flatten_callback_metrics(cm)
    assert out == pytest.approx({"train_loss": 0.5, "val_loss": 0.7})


def test_flatten_callback_metrics_empty():
    assert _flatten_callback_metrics({}) == {}


# ── _build_session_record ─────────────────────────────────────────────────

def _fake_trainer(metrics: dict, epochs: int = 1, steps: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        callback_metrics=metrics,
        current_epoch=epochs,
        global_step=steps,
    )


def test_session_record_basic_shape(tmp_path):
    session_dir = tmp_path / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    started = datetime(2026, 5, 11, 10, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 5, 11, 11, 0, tzinfo=timezone.utc)
    trainer = _fake_trainer(
        {"train_ce_epoch": torch.tensor(0.5), "val_ce_epoch": torch.tensor(0.7)},
        epochs=4, steps=128,
    )
    rec = _build_session_record(
        session_dir=session_dir,
        started_at=started, ended_at=ended,
        trainer=trainer,
        git_commit="abc123", git_dirty=False,
    )
    assert rec["session_name"] == "s001"
    assert rec["session_index"] == 1
    assert rec["started_at"] == started.isoformat()
    assert rec["ended_at"] == ended.isoformat()
    assert rec["epochs"] == 4
    assert rec["steps"] == 128
    assert rec["train_metric_key"] == "train_ce_epoch"
    assert rec["train_metric_value"] == pytest.approx(0.5)
    assert rec["val_metric_key"] == "val_ce_epoch"
    assert rec["val_metric_value"] == pytest.approx(0.7)
    assert rec["all_metrics"] == pytest.approx({"train_ce_epoch": 0.5, "val_ce_epoch": 0.7})
    assert rec["git_commit"] == "abc123"
    assert rec["git_dirty"] is False


def test_session_record_prefers_epoch_when_both_present(tmp_path):
    """A model logging both on_step and on_epoch surfaces the epoch value."""
    session_dir = tmp_path / "sessions" / "s002"
    session_dir.mkdir(parents=True)
    trainer = _fake_trainer({
        "train_ce_step": torch.tensor(5.0),    # last-batch loss
        "train_ce_epoch": torch.tensor(2.1),   # epoch average — what we want
        "val_ce_step": torch.tensor(4.0),
        "val_ce_epoch": torch.tensor(2.4),
    })
    rec = _build_session_record(
        session_dir=session_dir,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        trainer=trainer,
        git_commit=None, git_dirty=None,
    )
    assert rec["train_metric_key"] == "train_ce_epoch"
    assert rec["train_metric_value"] == pytest.approx(2.1)
    assert rec["val_metric_key"] == "val_ce_epoch"
    assert rec["val_metric_value"] == pytest.approx(2.4)


def test_session_record_falls_back_to_step_when_thats_all(tmp_path):
    """on_step=True only: surface _step value, with key preserved so the
    briefing can show `train_ce_step=...` (signaling "single batch")."""
    session_dir = tmp_path / "sessions" / "s003"
    session_dir.mkdir(parents=True)
    trainer = _fake_trainer({"train_ce_step": torch.tensor(0.4)})
    rec = _build_session_record(
        session_dir=session_dir,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        trainer=trainer,
        git_commit=None, git_dirty=None,
    )
    assert rec["train_metric_key"] == "train_ce_step"
    assert rec["train_metric_value"] == pytest.approx(0.4)


def test_session_record_missing_metrics(tmp_path):
    """No matching keys → metric fields absent, all_metrics still present."""
    session_dir = tmp_path / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    trainer = _fake_trainer({"accuracy": torch.tensor(0.95)})
    rec = _build_session_record(
        session_dir=session_dir,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        trainer=trainer,
        git_commit=None, git_dirty=None,
    )
    assert "train_metric_key" not in rec
    assert "val_metric_key" not in rec
    assert rec["all_metrics"] == pytest.approx({"accuracy": 0.95})


def test_session_record_index_parsed_from_name(tmp_path):
    """3-digit names parse correctly; so do 4+-digit overflow names."""
    for name, expected in (("s001", 1), ("s042", 42), ("s1000", 1000)):
        session_dir = tmp_path / "sessions" / name
        session_dir.mkdir(parents=True, exist_ok=True)
        trainer = _fake_trainer({})
        rec = _build_session_record(
            session_dir=session_dir,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            trainer=trainer,
            git_commit=None, git_dirty=None,
        )
        assert rec["session_index"] == expected


# ── _print_session_summary ────────────────────────────────────────────────

import json
from io import StringIO

from rich.console import Console


def _write_session(hset_dir: Path, session_name: str, **fields) -> None:
    """Append a single sessions.jsonl record."""
    hset_dir.mkdir(parents=True, exist_ok=True)
    record = {"session_name": session_name, **fields}
    with (hset_dir / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _capture_summary(hset_dir: Path, session_dir: Path) -> str:
    """Run _print_session_summary against a captured Console; return text."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    _print_session_summary(hset_dir, session_dir, console)
    return buf.getvalue()


def test_summary_emits_session_id_status_and_metrics(tmp_path):
    hset_dir = tmp_path / "h001"
    session_dir = hset_dir / "sessions" / "s002"
    session_dir.mkdir(parents=True)
    _write_session(
        hset_dir, "s002",
        status="completed", epochs=5, steps=400,
        train_metric_key="train_ce", train_metric_value=0.123,
        val_metric_key="val_ce", val_metric_value=0.456,
        est_sec_per_epoch=12.0,
    )
    out = _capture_summary(hset_dir, session_dir)
    assert "session s002" in out
    assert "done" in out
    assert "epochs=5" in out
    assert "steps=400" in out
    assert "train_ce=0.123" in out
    assert "val_ce=0.456" in out
    assert "12.0 sec/epoch" in out


def test_summary_interrupted_status_label(tmp_path):
    hset_dir = tmp_path / "h001"
    session_dir = hset_dir / "sessions" / "s003"
    session_dir.mkdir(parents=True)
    _write_session(hset_dir, "s003", status="interrupted", epochs=0, steps=12)
    out = _capture_summary(hset_dir, session_dir)
    assert "interrupted" in out
    assert "steps=12" in out


def test_summary_silent_when_record_doesnt_match_session(tmp_path):
    """If finalization didn't write a record for THIS session, print nothing
    — better silent than misleading."""
    hset_dir = tmp_path / "h001"
    session_dir = hset_dir / "sessions" / "s005"
    session_dir.mkdir(parents=True)
    # Record exists, but it's from an OLDER session.
    _write_session(hset_dir, "s004", status="completed", epochs=3)
    out = _capture_summary(hset_dir, session_dir)
    assert out == ""


def test_summary_silent_when_no_sessions_file(tmp_path):
    hset_dir = tmp_path / "h001"
    hset_dir.mkdir()
    session_dir = hset_dir / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    out = _capture_summary(hset_dir, session_dir)
    assert out == ""


def test_summary_with_loss_metric_keys(tmp_path):
    """Honor the _ce -> _loss transition: whatever key was recorded shows up."""
    hset_dir = tmp_path / "h001"
    session_dir = hset_dir / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    _write_session(
        hset_dir, "s001",
        status="completed", epochs=1, steps=10,
        train_metric_key="train_loss_epoch", train_metric_value=0.5,
        val_metric_key="val_loss_epoch", val_metric_value=0.6,
    )
    out = _capture_summary(hset_dir, session_dir)
    # _epoch suffix is stripped for display readability.
    assert "train_loss=0.500" in out
    assert "val_loss=0.600" in out
