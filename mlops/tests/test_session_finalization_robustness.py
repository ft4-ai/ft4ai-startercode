"""Robustness tests for SessionFinalizationCallback._finalize.

The session record is the only durable audit trail of "a fit() was attempted."
It must survive failures in metric rebuild, trainer-state queries, and
enrichment. Pre-fix, the existing _finalize wrapped everything in a single
swallow-all try/except, which meant a single failure deep in record assembly
silently lost the entire record. These tests pin the layered-finalization
design: build a minimal record first; write it; then best-effort enrich and
rebuild.

Backstory: stress test iter 8 surfaced this. A callback raised in
on_validation_epoch_end, fit() exited via exception, and the resulting
hset had a half-state with no sessions.jsonl. See bugs.md BUG-6.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest

from mlops.session_finalization import SessionFinalizationCallback


def _fake_trainer(metrics=None) -> SimpleNamespace:
    return SimpleNamespace(
        callback_metrics=metrics or {},
        current_epoch=1,
        global_step=10,
    )


def _make_hset(tmp_path: Path) -> tuple[Path, Path]:
    hset_dir = tmp_path / "hset"
    session_dir = hset_dir / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    return hset_dir, session_dir


def _setup_callback(hset_dir: Path, session_dir: Path) -> SessionFinalizationCallback:
    cb = SessionFinalizationCallback()
    cb.hset_dir = hset_dir
    cb.session_dir = session_dir
    cb.start(
        started_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        git_commit="deadbeef",
        git_dirty=False,
    )
    return cb


def _read_record(hset_dir: Path) -> dict:
    text = (hset_dir / "sessions.jsonl").read_text().strip()
    assert text, "sessions.jsonl exists but is empty"
    return json.loads(text.splitlines()[-1])


# ── Audit trail survives rebuild_cumulative_metrics failures ────────────────

def test_session_record_written_when_metrics_rebuild_fails(tmp_path, monkeypatch):
    """If rebuild_cumulative_metrics raises (e.g., partial CSV from a crashed
    session), the session record must still be written. The session record is
    the audit trail; cumulative metrics are a derived view.
    """
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)

    def boom(_):
        raise RuntimeError("simulated rebuild failure")
    monkeypatch.setattr(
        "mlops.session_finalization.rebuild_cumulative_metrics", boom,
    )

    cb.on_fit_end(_fake_trainer(), pl_module=None)

    record = _read_record(hset_dir)
    assert record["status"] == "completed"
    assert record["session_name"] == "s001"


# ── Audit trail survives trainer-state query failures ──────────────────────

def test_session_record_written_when_trainer_query_fails(tmp_path):
    """If reading trainer.callback_metrics or .current_epoch raises (e.g.,
    trainer is in a half-initialized state after an exception), a minimal
    record with status/timestamps/error must still be written.
    """
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    cb._status = "error"
    cb._error_info = "TestError: something went wrong"

    # Trainer where every attribute access raises.
    trainer = MagicMock()
    type(trainer).callback_metrics = PropertyMock(side_effect=RuntimeError("boom"))
    type(trainer).current_epoch = PropertyMock(side_effect=RuntimeError("boom"))
    type(trainer).global_step = PropertyMock(side_effect=RuntimeError("boom"))

    cb.teardown(trainer, pl_module=None, stage="fit")

    record = _read_record(hset_dir)
    assert record["status"] == "error"
    assert record["error"] == "TestError: something went wrong"
    assert record["session_name"] == "s001"
    assert "started_at" in record
    assert "ended_at" in record


# ── Audit trail survives both failures simultaneously ──────────────────────

def test_session_record_written_when_everything_post_minimal_fails(
    tmp_path, monkeypatch
):
    """Both enrichment and rebuild fail → minimal record still written."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    cb._status = "error"
    cb._error_info = "BadError: cascade"

    monkeypatch.setattr(
        "mlops.session_finalization.rebuild_cumulative_metrics",
        lambda _: (_ for _ in ()).throw(RuntimeError("rebuild failed")),
    )

    trainer = MagicMock()
    type(trainer).callback_metrics = PropertyMock(side_effect=RuntimeError("boom"))

    cb.teardown(trainer, pl_module=None, stage="fit")

    record = _read_record(hset_dir)
    assert record["status"] == "error"
    assert record["error"] == "BadError: cascade"


# ── hset.json's last_session_at is updated even on minimal-record write ───

def test_hset_last_session_at_updated_on_minimal_record(tmp_path, monkeypatch):
    """hset.json's last_session_at should reflect the most recent session
    attempt, regardless of whether enrichment succeeded."""
    hset_dir, session_dir = _make_hset(tmp_path)
    # Need an initial hset.json for update_hset_last_session to find.
    (hset_dir / "hset.json").write_text(json.dumps({
        "created_at": "2026-05-01T00:00:00+00:00",
        "last_session_at": "2026-05-01T00:00:00+00:00",
        "description": None,
        "config_hash": "abc",
    }))
    cb = _setup_callback(hset_dir, session_dir)

    monkeypatch.setattr(
        "mlops.session_finalization.rebuild_cumulative_metrics",
        lambda _: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    trainer = MagicMock()
    type(trainer).callback_metrics = PropertyMock(side_effect=RuntimeError("nope"))

    cb.on_fit_end(trainer, pl_module=None)

    data = json.loads((hset_dir / "hset.json").read_text())
    assert data["last_session_at"] != "2026-05-01T00:00:00+00:00"


# ── Diagnostic output: failures shouldn't be invisible ─────────────────────

def test_metrics_rebuild_failure_prints_warning(tmp_path, monkeypatch, capsys):
    """Swallowing exceptions is necessary (audit trail can't fail), but the
    user should know something went wrong. Pre-fix, failures were silent;
    post-fix, a warning goes to stderr.
    """
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)

    monkeypatch.setattr(
        "mlops.session_finalization.rebuild_cumulative_metrics",
        lambda _: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    cb.on_fit_end(_fake_trainer(), pl_module=None)

    captured = capsys.readouterr()
    # Stderr should mention the failure; phrasing is flexible.
    assert "metrics" in captured.err.lower() or "rebuild" in captured.err.lower()


# ── _finalize is still idempotent post-refactor ────────────────────────────

def test_finalize_idempotent_after_refactor(tmp_path):
    """on_fit_end + teardown on clean completion must not double-write."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.on_fit_end(trainer, pl_module=None)
    cb.teardown(trainer, pl_module=None, stage="fit")

    text = (hset_dir / "sessions.jsonl").read_text().strip()
    assert len(text.splitlines()) == 1
