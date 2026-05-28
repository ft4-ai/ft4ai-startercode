"""Session finalization callback.

Hooks into Lightning's natural exception-handling and end-of-fit machinery
to rebuild hset-level cumulative metrics and append a session record to
<hset>/sessions.jsonl. Runs whether fit() exits cleanly, is interrupted
via Ctrl-C, or crashes — via `on_exception`, `on_fit_end`, and `teardown`.
No user-code try/except needed; Lightning's own graceful-shutdown
sequence carries us.

Per-run state injection:
  - hset_dir, session_dir come from hset_state.inject_paths via the
    existing duck-typed attribute setting.
  - started_at, git_commit, git_dirty come from run.py via a direct
    .start(...) call right before trainer.fit().

The callback finalizes exactly once: `_finalized` guards against the
on_fit_end + teardown double-fire that happens on clean completion.

Layered finalization (the audit trail must survive partial failures):
  Stage 1: build a MINIMAL record from callback state only. This always
           succeeds — no trainer queries, no derived data.
  Stage 2: best-effort enrichment from trainer state (epochs, steps,
           final losses, all metrics). Failure → warning + minimal record.
  Stage 3a: append the session record. This is the audit trail; must succeed.
  Stage 3b: best-effort update of hset.json's last_session_at (a derived
           view that powers stats display). Failure → warning.
  Stage 4: best-effort cumulative metrics rebuild. Failure → warning;
           per-session CSVs remain authoritative.

Also tracks step-end timestamps so the briefing can extrapolate a stable
"sec/epoch" estimate from the recent step rate (robust to warmup, sleep
gaps, and partial epochs — naive `total_wall / epochs` is not).
"""
from __future__ import annotations

import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightning as L

from mlops.cumulative_metrics import rebuild_cumulative_metrics
from mlops.hset_state import append_session, update_hset_last_session


# How many recent step-end timestamps to keep for sec/epoch extrapolation.
# Long enough to smooth noise; short enough that one outlier (laptop sleep,
# eviction pause) survives the median pool.
_STEP_TIMESTAMP_WINDOW = 50


class SessionFinalizationCallback(L.Callback):
    """At end of fit(), rebuild cumulative metrics and append a session
    record. Status reflects how fit() exited: 'completed', 'interrupted',
    or 'error'.
    """

    def __init__(self) -> None:
        super().__init__()
        # Injected by hset_state.inject_paths (duck-typed):
        self.hset_dir: Optional[Path] = None
        self.session_dir: Optional[Path] = None
        # Injected by run.py via .start(...):
        self.started_at: Optional[datetime] = None
        self.git_commit: Optional[str] = None
        self.git_dirty: Optional[bool] = None
        self.rdesc: Optional[str] = None
        # State set when hooks fire:
        self._status = "completed"
        self._error_info: Optional[str] = None
        self._finalized = False
        # Ring buffer of recent `time.monotonic()` values at each
        # on_train_batch_end. Used for sec/epoch extrapolation.
        self._step_timestamps: deque[float] = deque(maxlen=_STEP_TIMESTAMP_WINDOW)

    def start(self, started_at: datetime, git_commit: Optional[str],
              git_dirty: Optional[bool], rdesc: Optional[str] = None) -> None:
        """Called by run.py immediately before trainer.fit()."""
        self.started_at = started_at
        self.git_commit = git_commit
        self.git_dirty = git_dirty
        self.rdesc = rdesc

    def on_train_batch_end(  # type: ignore[override]
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self._step_timestamps.append(time.monotonic())

    def on_exception(self, trainer, pl_module, exception):  # type: ignore[override]
        if isinstance(exception, KeyboardInterrupt):
            self._status = "interrupted"
        else:
            self._status = "error"
            self._error_info = f"{type(exception).__name__}: {exception}"
        # Lightning's trainer/call.py skips _call_teardown_hook on both
        # KeyboardInterrupt and general BaseException paths, so neither
        # our teardown() nor on_fit_end fires for non-clean exits.
        # on_exception is the only hook we can rely on here — finalize from
        # inside it. The try/except prevents a finalization crash from
        # escaping Lightning's shutdown sequence (which would otherwise turn
        # a clean Ctrl-C into a noisy traceback because Lightning's
        # sys.exit(1) never runs if _interrupt re-raises).
        try:
            self._finalize(trainer)
        except Exception as e:
            print(
                f"Warning: session finalization on {self._status} failed "
                f"({type(e).__name__}: {e}); sessions.jsonl may be missing this run.",
                file=sys.stderr,
            )

    def on_fit_end(self, trainer, pl_module):  # type: ignore[override]
        self._finalize(trainer)

    def teardown(self, trainer, pl_module, stage):  # type: ignore[override]
        # Safety net for the rare case where on_fit_end doesn't fire (e.g.,
        # a setup-phase error that aborts before training). Guarded by
        # _finalized so on_fit_end + teardown on clean completion doesn't
        # double-write.
        if stage == "fit":
            self._finalize(trainer)

    # ── Finalization (layered: each stage independent) ────────────────────

    def _finalize(self, trainer) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Skip if we weren't fully wired up (e.g., setup error before
        # run.py's .start() call). Nothing meaningful to record.
        if (self.hset_dir is None or self.session_dir is None
                or self.started_at is None):
            return

        ended_at = datetime.now(timezone.utc)

        # Stage 1: minimal record. Built from callback state only —
        # guaranteed to succeed barring filesystem-level catastrophe.
        record = self._minimal_record(ended_at)

        # Stage 2: best-effort enrichment from trainer state. May fail
        # if trainer is in a half-state after an in-fit exception.
        try:
            from mlops.run import _build_session_record_enrichment
            record.update(_build_session_record_enrichment(trainer))
        except Exception as e:
            print(
                f"Warning: session record enrichment failed "
                f"({type(e).__name__}: {e}); writing minimal record.",
                file=sys.stderr,
            )

        # Stage 2b: best-effort step-rate extrapolation. Folded in here so
        # a half-state trainer doesn't drop the rate, which we tracked
        # ourselves from on_train_batch_end.
        try:
            record.update(self._step_rate_fields(trainer))
        except Exception as e:
            print(
                f"Warning: step-rate computation failed "
                f"({type(e).__name__}: {e}); briefing will fall back to "
                f"total-wall timing.",
                file=sys.stderr,
            )

        # Stage 3a: append the session record. This is the audit trail
        # and must succeed.
        append_session(self.hset_dir, record)

        # Stage 3b: update hset.json's last_session_at — a derived view
        # for stats display. The source of truth for "when was the last
        # session" is sessions.jsonl's newest record; this is just a cache.
        try:
            update_hset_last_session(self.hset_dir, ended_at)
        except Exception as e:
            print(
                f"Warning: hset.json last_session_at update failed "
                f"({type(e).__name__}: {e}); sessions.jsonl unaffected.",
                file=sys.stderr,
            )

        # Stage 4: best-effort cumulative metrics rebuild. Per-session
        # CSVs are authoritative; the hset-level rebuild is a derived view.
        try:
            rebuild_cumulative_metrics(self.hset_dir)
        except Exception as e:
            print(
                f"Warning: cumulative metrics rebuild failed "
                f"({type(e).__name__}: {e}); per-session CSVs are unaffected.",
                file=sys.stderr,
            )

    def _step_rate_fields(self, trainer) -> dict:
        """Compute sec/epoch extrapolation from recent step timestamps.

        Median of inter-step deltas keeps one outlier (laptop sleep, GC
        pause, eviction) from poisoning the estimate. If we don't have at
        least 3 deltas, return an empty dict (the briefing will fall back
        to total-wall timing).
        """
        if len(self._step_timestamps) < 4:
            return {}
        deltas = [
            self._step_timestamps[i] - self._step_timestamps[i - 1]
            for i in range(1, len(self._step_timestamps))
        ]
        recent_sec_per_step = statistics.median(deltas)
        out = {"recent_sec_per_step": recent_sec_per_step}
        steps_per_epoch = getattr(trainer, "num_training_batches", None)
        if isinstance(steps_per_epoch, int) and steps_per_epoch > 0:
            out["steps_per_epoch"] = steps_per_epoch
            out["est_sec_per_epoch"] = recent_sec_per_step * steps_per_epoch
        return out

    def _minimal_record(self, ended_at: datetime) -> dict:
        """Session-record fields drawn entirely from callback state.

        These are the fields the audit trail can't lose even if the trainer
        is unreachable: when did we start, when did we end, how did fit()
        exit, what was the exception (if any), and the provenance bits.
        """
        record = {
            "session_name": self.session_dir.name,
            "session_index": int(self.session_dir.name[1:]),
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "status": self._status,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }
        if self._error_info is not None:
            record["error"] = self._error_info
        if self.rdesc is not None:
            record["rdesc"] = self.rdesc
        return record
