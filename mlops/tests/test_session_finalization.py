"""Unit tests for SessionFinalizationCallback.

Tests the three exit paths (clean / KeyboardInterrupt / other exception)
by directly invoking the callback hooks. The end-to-end clean path is also
exercised by tests/integration/test_run_smart_flow.py.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlops.session_finalization import SessionFinalizationCallback


def _fake_trainer(metrics=None) -> SimpleNamespace:
    """Minimal stand-in for trainer; only needs the attributes the
    callback and _build_session_record read."""
    return SimpleNamespace(
        callback_metrics=metrics or {},
        current_epoch=1,
        global_step=10,
    )


def _make_hset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a hset dir with sessions/s001 ready for the callback to record into."""
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
        git_commit=None,
        git_dirty=None,
    )
    return cb


def _read_session_record(hset_dir: Path) -> dict:
    text = (hset_dir / "sessions.jsonl").read_text().strip()
    assert text, "sessions.jsonl empty"
    lines = text.splitlines()
    return json.loads(lines[-1])


# ── Clean completion ──────────────────────────────────────────────────────

def test_on_fit_end_writes_completed_record(tmp_path):
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer({"train_loss": torch.tensor(0.5)})

    cb.on_fit_end(trainer, pl_module=None)

    record = _read_session_record(hset_dir)
    assert record["status"] == "completed"
    assert "error" not in record
    assert record["session_name"] == "s001"


# ── KeyboardInterrupt ─────────────────────────────────────────────────────

def test_keyboard_interrupt_records_interrupted_status(tmp_path):
    """Lightning's flow on Ctrl-C skips _call_teardown_hook entirely
    (trainer/call.py:57-66), so on_exception is the only hook we can rely on.
    The callback must finalize directly inside on_exception."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.on_exception(trainer, pl_module=None, exception=KeyboardInterrupt())
    # Deliberately DO NOT call teardown — Lightning doesn't on Ctrl-C.

    record = _read_session_record(hset_dir)
    assert record["status"] == "interrupted"
    assert "error" not in record  # Ctrl-C doesn't carry an error message


# ── Other exception ───────────────────────────────────────────────────────

def test_other_exception_records_error_with_info(tmp_path):
    """Same path as KeyboardInterrupt — Lightning's BaseException branch
    (call.py:68-73) also skips _call_teardown_hook. on_exception must
    finalize on its own."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.on_exception(trainer, pl_module=None,
                    exception=RuntimeError("CUDA out of memory"))
    # Deliberately DO NOT call teardown.

    record = _read_session_record(hset_dir)
    assert record["status"] == "error"
    assert record["error"] == "RuntimeError: CUDA out of memory"


def test_on_exception_then_teardown_does_not_double_write(tmp_path):
    """Defensive: if Lightning ever DOES call teardown after on_exception
    (e.g., for the TunerExitException path), the _finalized guard prevents
    a duplicate record."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.on_exception(trainer, pl_module=None, exception=KeyboardInterrupt())
    cb.teardown(trainer, pl_module=None, stage="fit")

    lines = (hset_dir / "sessions.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_on_exception_swallows_finalize_crash(tmp_path, monkeypatch, capsys):
    """If append_session itself crashes (filesystem full, permission denied),
    the exception MUST NOT escape on_exception — otherwise Lightning's
    sys.exit(1) never runs and the user sees a confusing traceback instead
    of a graceful Ctrl-C exit."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr("mlops.session_finalization.append_session", boom)

    # Must not raise.
    cb.on_exception(trainer, pl_module=None, exception=KeyboardInterrupt())

    assert "Warning: session finalization on interrupted failed" in capsys.readouterr().err


# ── Idempotence: clean completion fires on_fit_end + teardown ─────────────

def test_finalization_runs_exactly_once_on_clean_completion(tmp_path):
    """Lightning fires both on_fit_end AND teardown on clean completion.
    The callback must record only once."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.on_fit_end(trainer, pl_module=None)
    cb.teardown(trainer, pl_module=None, stage="fit")

    sessions = (hset_dir / "sessions.jsonl").read_text().splitlines()
    assert len(sessions) == 1


# ── Teardown for non-fit stages doesn't record ────────────────────────────

def test_teardown_for_validate_stage_does_nothing(tmp_path):
    """Lightning calls teardown for fit/validate/test/predict; we only
    finalize for the 'fit' stage."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()

    cb.teardown(trainer, pl_module=None, stage="validate")

    assert not (hset_dir / "sessions.jsonl").exists()


# ── Not-yet-wired-up callback no-ops ─────────────────────────────────────

def test_finalize_without_injection_is_silent(tmp_path):
    """If the callback never had .start() called (setup-phase failure),
    on_fit_end/teardown should no-op rather than crash."""
    cb = SessionFinalizationCallback()
    cb.hset_dir = tmp_path  # injected partially
    # session_dir, started_at, etc. NOT set
    trainer = _fake_trainer()

    cb.on_fit_end(trainer, pl_module=None)

    # No sessions.jsonl created.
    assert not (tmp_path / "sessions.jsonl").exists()


# ── Cumulative metrics rebuilt on finalize ────────────────────────────────

# ── Step-rate tracking → est_sec_per_epoch on the session record ────────

def test_step_rate_recorded_when_enough_samples(tmp_path):
    """on_train_batch_end timestamps yield a median-based sec/epoch estimate."""
    import time
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)

    # Synthesize 20 evenly-spaced step timestamps at ~0.1 sec each.
    t0 = time.monotonic()
    cb._step_timestamps.clear()
    for i in range(20):
        cb._step_timestamps.append(t0 + i * 0.1)

    trainer = SimpleNamespace(
        callback_metrics={},
        current_epoch=1,
        global_step=20,
        num_training_batches=50,
    )
    cb.on_fit_end(trainer, pl_module=None)

    rec = _read_session_record(hset_dir)
    assert rec["recent_sec_per_step"] == pytest.approx(0.1, rel=0.05)
    assert rec["steps_per_epoch"] == 50
    assert rec["est_sec_per_epoch"] == pytest.approx(5.0, rel=0.05)


def test_step_rate_robust_to_outlier(tmp_path):
    """One huge delta (e.g., laptop sleep) is ignored by the median."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)

    # 19 fast steps + 1 huge gap.
    cb._step_timestamps.clear()
    t = 0.0
    cb._step_timestamps.append(t)
    for i in range(18):
        t += 0.1
        cb._step_timestamps.append(t)
    # Then a 3600-second sleep:
    t += 3600
    cb._step_timestamps.append(t)

    trainer = SimpleNamespace(
        callback_metrics={},
        current_epoch=1,
        global_step=20,
        num_training_batches=50,
    )
    cb.on_fit_end(trainer, pl_module=None)
    rec = _read_session_record(hset_dir)
    # Median over 19 deltas: 18 are 0.1, one is 3600. Median is 0.1.
    assert rec["recent_sec_per_step"] == pytest.approx(0.1, rel=0.05)


def test_step_rate_omitted_when_too_few_samples(tmp_path):
    """<4 timestamps → fields are absent, briefing falls back to wall-clock."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    # Only 2 timestamps captured (e.g. user Ctrl-C'd very early).
    cb._step_timestamps.clear()
    cb._step_timestamps.append(0.0)
    cb._step_timestamps.append(0.1)

    trainer = SimpleNamespace(
        callback_metrics={}, current_epoch=0, global_step=2,
        num_training_batches=50,
    )
    cb.on_fit_end(trainer, pl_module=None)
    rec = _read_session_record(hset_dir)
    assert "est_sec_per_epoch" not in rec
    assert "recent_sec_per_step" not in rec


def test_finalize_rebuilds_cumulative_metrics(tmp_path):
    hset_dir, session_dir = _make_hset(tmp_path)
    # Write a fake session metrics.csv that rebuild can pick up.
    (session_dir / "metrics.csv").write_text("step,loss\n0,0.5\n1,0.4\n")

    cb = _setup_callback(hset_dir, session_dir)
    trainer = _fake_trainer()
    cb.on_fit_end(trainer, pl_module=None)

    cumulative = hset_dir / "metrics.csv"
    assert cumulative.is_file()
    text = cumulative.read_text()
    assert "session" in text  # leading column
    assert "0.5" in text       # row from session


def test_rdesc_persisted_into_session_record(tmp_path):
    """When run.py passes rdesc via .start(), it lands in sessions.jsonl."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = SessionFinalizationCallback()
    cb.hset_dir = hset_dir
    cb.session_dir = session_dir
    cb.start(
        started_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        git_commit=None,
        git_dirty=None,
        rdesc="warmup + cosine, lr=3e-4",
    )
    cb.on_fit_end(_fake_trainer(), pl_module=None)
    rec = _read_session_record(hset_dir)
    assert rec["rdesc"] == "warmup + cosine, lr=3e-4"


def test_rdesc_omitted_when_not_set(tmp_path):
    """Backward-compatible default: no rdesc key when the user didn't pass --rdesc."""
    hset_dir, session_dir = _make_hset(tmp_path)
    cb = _setup_callback(hset_dir, session_dir)
    cb.on_fit_end(_fake_trainer(), pl_module=None)
    rec = _read_session_record(hset_dir)
    assert "rdesc" not in rec
