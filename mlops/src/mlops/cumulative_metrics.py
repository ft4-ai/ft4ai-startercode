"""Concatenate per-session metrics into a unified hset-level CSV.

Each training session writes its own `<hset>/sessions/sNNN/metrics.csv`
via stock Lightning CSVLogger. After each session ends — clean, interrupted,
or errored — SessionFinalizationCallback calls `rebuild_cumulative_metrics`
to merge every session file into a single `<hset>/metrics.csv`.

That's it. No live-update callback, no Lightning coupling at the module
level. Schema evolution between sessions is handled by outer-joining
columns: rows from a session that didn't log column X get an empty cell
for X. Session order matches lexicographic order of the zero-padded sNNN
directory names, which coincides with creation order.

A `session` column (integer index, not the sNNN string) is prepended to
each row so downstream tooling can filter/group by session.

Live monitoring during a session is provided by Lightning's progress bar
and by Lightning's own session-level CSVLogger flushes — no work for ft4
to do.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

_SESSION_RE = re.compile(r"^s(\d{3,})$")


def rebuild_cumulative_metrics(hset_dir: Path) -> Path | None:
    """Read all per-session metrics.csv files under <hset>/sessions/ and
    write a unified <hset>/metrics.csv.

    Returns the cumulative file path on success, or None if no session
    data exists yet (no sessions/ dir, no sNNN dirs, or all session files
    are empty).

    Atomic: writes to <hset>/metrics.csv.tmp first, then renames. A
    crashed run never leaves a partially-written cumulative file.
    """
    sessions_root = hset_dir / "sessions"
    if not sessions_root.is_dir():
        return None

    session_dirs = sorted(
        (p for p in sessions_root.iterdir()
         if p.is_dir() and _SESSION_RE.match(p.name)),
        key=lambda p: p.name,
    )

    all_columns: list[str] = ["session"]
    all_rows: list[dict] = []
    for session_dir in session_dirs:
        metrics_file = session_dir / "metrics.csv"
        if not metrics_file.is_file():
            continue
        session_idx = int(_SESSION_RE.match(session_dir.name).group(1))
        with metrics_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for col in (reader.fieldnames or []):
                if col not in all_columns:
                    all_columns.append(col)
            for row in reader:
                all_rows.append({"session": session_idx, **row})

    if not all_rows:
        return None

    cumulative = hset_dir / "metrics.csv"
    tmp = cumulative.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    tmp.replace(cumulative)
    return cumulative
