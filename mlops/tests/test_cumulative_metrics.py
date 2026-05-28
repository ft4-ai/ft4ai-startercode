"""Unit tests for cumulative_metrics."""
import csv
from pathlib import Path

from mlops.cumulative_metrics import rebuild_cumulative_metrics


def _write_session_metrics(
    hset_dir: Path, session_name: str, header: list[str], rows: list[list]
) -> Path:
    """Helper: create <hset>/sessions/<session_name>/metrics.csv with the
    given content. Returns the file path."""
    session_dir = hset_dir / "sessions" / session_name
    session_dir.mkdir(parents=True)
    metrics_file = session_dir / "metrics.csv"
    with metrics_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return metrics_file


def _read_cumulative(hset_dir: Path) -> tuple[list[str], list[dict]]:
    path = hset_dir / "metrics.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


# ── No-op cases ────────────────────────────────────────────────────────────

def test_no_sessions_dir_returns_none(tmp_path):
    assert rebuild_cumulative_metrics(tmp_path) is None
    assert not (tmp_path / "metrics.csv").exists()


def test_empty_sessions_dir_returns_none(tmp_path):
    (tmp_path / "sessions").mkdir()
    assert rebuild_cumulative_metrics(tmp_path) is None
    assert not (tmp_path / "metrics.csv").exists()


def test_session_dir_with_no_metrics_file_returns_none(tmp_path):
    (tmp_path / "sessions" / "s001").mkdir(parents=True)
    assert rebuild_cumulative_metrics(tmp_path) is None


def test_header_only_file_returns_none(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [])
    assert rebuild_cumulative_metrics(tmp_path) is None


# ── Single session ─────────────────────────────────────────────────────────

def test_single_session_writes_cumulative(tmp_path):
    _write_session_metrics(
        tmp_path, "s001",
        ["step", "loss"],
        [[0, 0.5], [1, 0.4]],
    )
    result = rebuild_cumulative_metrics(tmp_path)
    assert result == tmp_path / "metrics.csv"
    fieldnames, rows = _read_cumulative(tmp_path)
    assert fieldnames == ["session", "step", "loss"]
    assert len(rows) == 2
    assert all(r["session"] == "1" for r in rows)


# ── Multiple sessions, same schema ────────────────────────────────────────

def test_two_sessions_same_schema_concatenated(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5], [1, 0.4]])
    _write_session_metrics(tmp_path, "s002", ["step", "loss"], [[2, 0.3], [3, 0.2]])
    rebuild_cumulative_metrics(tmp_path)
    fieldnames, rows = _read_cumulative(tmp_path)
    assert fieldnames == ["session", "step", "loss"]
    assert len(rows) == 4
    assert [r["session"] for r in rows] == ["1", "1", "2", "2"]
    assert [r["step"] for r in rows] == ["0", "1", "2", "3"]


def test_session_ordering_is_lexicographic(tmp_path):
    """Zero-padded names sort correctly even with >=10 sessions."""
    _write_session_metrics(tmp_path, "s002", ["step"], [[10]])
    _write_session_metrics(tmp_path, "s010", ["step"], [[100]])
    _write_session_metrics(tmp_path, "s001", ["step"], [[1]])
    rebuild_cumulative_metrics(tmp_path)
    _, rows = _read_cumulative(tmp_path)
    assert [r["session"] for r in rows] == ["1", "2", "10"]
    assert [r["step"] for r in rows] == ["1", "10", "100"]


# ── Schema evolution ──────────────────────────────────────────────────────

def test_new_column_in_later_session(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5]])
    _write_session_metrics(
        tmp_path, "s002",
        ["step", "loss", "accuracy"],
        [[1, 0.3, 0.9]],
    )
    rebuild_cumulative_metrics(tmp_path)
    fieldnames, rows = _read_cumulative(tmp_path)
    assert "accuracy" in fieldnames
    assert len(rows) == 2
    # Session 1's row has empty accuracy.
    s1 = [r for r in rows if r["session"] == "1"][0]
    assert s1["accuracy"] == ""
    s2 = [r for r in rows if r["session"] == "2"][0]
    assert s2["accuracy"] == "0.9"


def test_dropped_column_in_later_session(tmp_path):
    """Session 2 has fewer columns than session 1; session 2's row gets
    empty for the dropped column."""
    _write_session_metrics(tmp_path, "s001", ["step", "loss", "extra"], [[0, 0.5, 99]])
    _write_session_metrics(tmp_path, "s002", ["step", "loss"], [[1, 0.3]])
    rebuild_cumulative_metrics(tmp_path)
    fieldnames, rows = _read_cumulative(tmp_path)
    assert "extra" in fieldnames
    s2 = [r for r in rows if r["session"] == "2"][0]
    assert s2["extra"] == ""


def test_column_order_preserves_first_occurrence(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5]])
    _write_session_metrics(
        tmp_path, "s002",
        ["step", "loss", "accuracy"],
        [[1, 0.3, 0.9]],
    )
    rebuild_cumulative_metrics(tmp_path)
    fieldnames, _ = _read_cumulative(tmp_path)
    # session always first; then step, loss (from s001); then accuracy (added in s002).
    assert fieldnames == ["session", "step", "loss", "accuracy"]


# ── Re-running (atomic replacement) ───────────────────────────────────────

def test_rerun_replaces_old_cumulative(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5]])
    rebuild_cumulative_metrics(tmp_path)
    fieldnames1, rows1 = _read_cumulative(tmp_path)
    assert len(rows1) == 1

    # Add a new session and rerun.
    _write_session_metrics(tmp_path, "s002", ["step", "loss"], [[1, 0.3]])
    rebuild_cumulative_metrics(tmp_path)

    fieldnames2, rows2 = _read_cumulative(tmp_path)
    assert len(rows2) == 2
    # The old single-row file isn't a leftover — fully replaced.
    assert [r["session"] for r in rows2] == ["1", "2"]


def test_no_tmp_file_left_behind(tmp_path):
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5]])
    rebuild_cumulative_metrics(tmp_path)
    # The .tmp staging file from the atomic write must not leak.
    assert not (tmp_path / "metrics.csv.tmp").exists()


# ── Robustness ────────────────────────────────────────────────────────────

def test_ignores_non_sNNN_dirs(tmp_path):
    """Stray subdirectories with non-matching names are skipped silently."""
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "junk").mkdir()
    (tmp_path / "sessions" / "junk" / "metrics.csv").write_text("step,loss\n0,99\n")
    _write_session_metrics(tmp_path, "s001", ["step", "loss"], [[0, 0.5]])
    rebuild_cumulative_metrics(tmp_path)
    _, rows = _read_cumulative(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session"] == "1"
