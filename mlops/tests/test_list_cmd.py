"""Unit tests for mlops.list_cmd.

Tests are filesystem-fixture style: build a fake `runs/<stem>/` tree with
hset.json / sessions.jsonl / config.yaml / samples/, then assert on the
Row collection and rendering.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from mlops.hset_state import Ft4Args
from mlops.list_cmd import (
    Row,
    _collect_rows,
    _compute_sample_budget,
    _FIXED_COL_WIDTHS,
    _SAMPLES_MIN_WIDTH,
    _format_steps,
    _format_val,
    _last_status,
    _latest_sample_line,
    _pick_val_label,
    _relative_time,
    _render_plain,
    _render_rich,
    _sort_rows,
    run_list,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_hset(
    version_dir: Path,
    name: str,
    *,
    config: dict | None = None,
    sessions: list[dict] | None = None,
    description: str | None = None,
    last_session_at: str | None = None,
    samples: dict[int, str] | None = None,
) -> Path:
    """Build a fake hset under `version_dir`."""
    hset_dir = version_dir / name
    hset_dir.mkdir(parents=True)
    hset_json = {
        "config_hash": "ch-" + name,
        "description": description,
    }
    if last_session_at is not None:
        hset_json["last_session_at"] = last_session_at
    (hset_dir / "hset.json").write_text(json.dumps(hset_json))
    if config is not None:
        (hset_dir / "config.yaml").write_text(yaml.safe_dump(config))
    if sessions:
        with (hset_dir / "sessions.jsonl").open("w") as f:
            for s in sessions:
                f.write(json.dumps(s) + "\n")
    if samples:
        sdir = hset_dir / "samples"
        sdir.mkdir()
        for step, text in samples.items():
            (sdir / f"step_{step:06d}.md").write_text(text)
    return hset_dir


def _make_version(model_dir: Path, name: str) -> Path:
    v = model_dir / name
    v.mkdir(parents=True)
    (v / "version.json").write_text(json.dumps({
        "state_dict_hash": "sd-" + name,
        "model_file_hash": "mf-" + name,
    }))
    return v


def _args_for_list(model_file: Path, **kw) -> Ft4Args:
    return Ft4Args(
        subcommand="list",
        model_file=model_file,
        sort_by=kw.pop("sort_by", "time"),
        limit=kw.pop("limit", None),
        version=kw.pop("version", None),
        **kw,
    )


# ── _relative_time ─────────────────────────────────────────────────────────

def test_relative_time_just_now():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso = (now - timedelta(seconds=30)).isoformat()
    assert _relative_time(iso, now) == "just now"


def test_relative_time_minutes():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso = (now - timedelta(minutes=5)).isoformat()
    assert _relative_time(iso, now) == "5m ago"


def test_relative_time_hours():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso = (now - timedelta(hours=3)).isoformat()
    assert _relative_time(iso, now) == "3h ago"


def test_relative_time_days():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso = (now - timedelta(days=2)).isoformat()
    assert _relative_time(iso, now) == "2d ago"


def test_relative_time_future_clamped():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso = (now + timedelta(seconds=5)).isoformat()
    assert _relative_time(iso, now) == "just now"


def test_relative_time_invalid():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_time("not-a-date", now) == "—"


# ── _latest_sample_line ────────────────────────────────────────────────────

def test_latest_sample_line_picks_highest_step(tmp_path):
    hset_dir = tmp_path / "h001"
    hset_dir.mkdir()
    sdir = hset_dir / "samples"
    sdir.mkdir()
    (sdir / "step_000001.md").write_text(
        "**Prompt:** `old`\n\nold completion-old\n\n---\n"
    )
    (sdir / "step_000100.md").write_text(
        "**Prompt:** `new`\n\nnew completion-new\n\n---\n"
    )
    out = _latest_sample_line(hset_dir)
    assert out.startswith("new")
    assert "completion-new" in out


def test_latest_sample_line_no_samples(tmp_path):
    hset_dir = tmp_path / "h001"
    hset_dir.mkdir()
    assert _latest_sample_line(hset_dir) == ""


def test_latest_sample_line_collapses_newlines(tmp_path):
    hset_dir = tmp_path / "h001"
    hset_dir.mkdir()
    sdir = hset_dir / "samples"
    sdir.mkdir()
    (sdir / "step_000005.md").write_text(
        "**Prompt:** `hello`\n\nhello world\nwith\nnewlines\n\n---\n"
    )
    out = _latest_sample_line(hset_dir)
    assert "\n" not in out


# ── _format_steps / _format_val ────────────────────────────────────────────

def test_format_steps():
    assert _format_steps(0) == "0"
    assert _format_steps(999) == "999"
    assert _format_steps(1_500) == "2k"
    assert _format_steps(251_000) == "251k"
    assert _format_steps(1_200_000) == "1.2M"


def test_format_val_none():
    assert _format_val(None) == "—"


def test_format_val_float():
    assert _format_val(0.857) == "0.857"
    assert _format_val(1.6845) == "1.685"


# ── _last_status ───────────────────────────────────────────────────────────

def test_last_status_compacts():
    assert _last_status([]) == "—"
    assert _last_status([{"status": "completed"}]) == "done"
    assert _last_status([{"status": "interrupted"}]) == "intrpt"
    assert _last_status([{"status": "error"}]) == "error"
    # Picks the last session's status.
    assert _last_status([
        {"status": "completed"}, {"status": "interrupted"},
    ]) == "intrpt"


# ── _pick_val_label ────────────────────────────────────────────────────────

def test_pick_val_label_majority(tmp_path):
    rows = [
        _row(val_label="val_ce"),
        _row(val_label="val_ce"),
        _row(val_label="val_loss"),
    ]
    assert _pick_val_label(rows) == "val_ce"


def test_pick_val_label_fallback_no_labels():
    rows = [_row(val_label="")]
    assert _pick_val_label(rows) == "val"


def _row(**overrides) -> Row:
    base = dict(
        slug="v001/h001",
        notes="",
        sessions=0,
        last_at="—",
        last_at_secs=0.0,
        val_label="",
        val_value=None,
        steps=0,
        status="—",
        sample_line="",
    )
    base.update(overrides)
    return Row(**base)


# ── _collect_rows + _sort_rows (integration on fixture tree) ───────────────

def _build_model_dir(tmp_path: Path) -> Path:
    """Build a runs/iris tree:
        v001/h001  baseline, val_ce=0.857, 8 steps, 2d ago
        v001/h002  smaller batch, val_ce=0.340, 45 steps, 1h ago
        v002/h001  no description, val_ce=0.500, 16 steps, just now
    """
    model_dir = tmp_path / "iris"
    v1 = _make_version(model_dir, "v001")
    now = datetime.now(timezone.utc)
    _make_hset(
        v1, "h001",
        description="baseline",
        last_session_at=(now - timedelta(days=2)).isoformat(),
        config={"model": {"dim": 8}, "data": {"batch_size": 32}},
        sessions=[{
            "status": "completed", "steps": 8,
            "val_metric_key": "val_ce", "val_metric_value": 0.857,
        }],
    )
    _make_hset(
        v1, "h002",
        description="smaller batch",
        last_session_at=(now - timedelta(hours=1)).isoformat(),
        config={"model": {"dim": 8}, "data": {"batch_size": 8}},
        sessions=[{
            "status": "completed", "steps": 45,
            "val_metric_key": "val_ce", "val_metric_value": 0.340,
        }],
    )
    v2 = _make_version(model_dir, "v002")
    _make_hset(
        v2, "h001",
        last_session_at=now.isoformat(),
        config={"model": {"dim": 16}, "data": {"batch_size": 32}},
        sessions=[{
            "status": "completed", "steps": 16,
            "val_metric_key": "val_ce", "val_metric_value": 0.500,
        }],
    )
    return model_dir


def test_collect_rows_basic(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    rows = _collect_rows(model_dir, version_filter=None)
    assert {r.slug for r in rows} == {"v001/h001", "v001/h002", "v002/h001"}
    by_slug = {r.slug: r for r in rows}
    assert by_slug["v001/h001"].notes == "baseline"
    assert by_slug["v001/h002"].notes == "smaller batch"
    # v002/h001 has no description and no baseline-diff target available
    # since it's h001 of its version → empty notes.
    assert by_slug["v002/h001"].notes == ""


def test_collect_rows_version_filter(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    rows = _collect_rows(model_dir, version_filter="v001")
    assert {r.slug for r in rows} == {"v001/h001", "v001/h002"}


def test_sort_rows_by_time_newest_first(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    rows = _sort_rows(_collect_rows(model_dir, version_filter=None), "time")
    assert [r.slug for r in rows] == ["v002/h001", "v001/h002", "v001/h001"]


def test_sort_rows_by_perf_smallest_val_first(tmp_path):
    model_dir = _build_model_dir(tmp_path)
    rows = _sort_rows(_collect_rows(model_dir, version_filter=None), "perf")
    assert [r.slug for r in rows] == ["v001/h002", "v002/h001", "v001/h001"]


def test_collect_row_notes_falls_back_to_diff(tmp_path):
    """When hdesc isn't set, Notes shows the config diff vs baseline (h001).
    Uses `model.dim` rather than `data.batch_size` since the latter is now
    in the exclusion list — flipping batch_size doesn't show in diffs."""
    model_dir = tmp_path / "iris"
    v = _make_version(model_dir, "v001")
    _make_hset(
        v, "h001",
        config={"model": {"dim": 8}, "trainer": {"gradient_clip_val": 1.0}},
        sessions=[{"status": "completed", "steps": 8}],
        last_session_at=datetime.now(timezone.utc).isoformat(),
    )
    _make_hset(
        v, "h002",  # no description
        config={"model": {"dim": 16}, "trainer": {"gradient_clip_val": 1.0}},
        sessions=[{"status": "completed", "steps": 16}],
        last_session_at=datetime.now(timezone.utc).isoformat(),
    )
    rows = _collect_rows(model_dir, version_filter=None)
    by_slug = {r.slug: r for r in rows}
    assert "model.dim=16" in by_slug["v001/h002"].notes


def test_collect_row_notes_skips_excluded_keys(tmp_path):
    """Changing only excluded keys (seed_everything, data.batch_size, etc.)
    should produce empty Notes — those aren't scientifically meaningful."""
    model_dir = tmp_path / "iris"
    v = _make_version(model_dir, "v001")
    _make_hset(
        v, "h001",
        config={"model": {"dim": 8}, "data": {"init_args": {"batch_size": 32}}, "seed_everything": 1},
        sessions=[{"status": "completed", "steps": 8}],
        last_session_at=datetime.now(timezone.utc).isoformat(),
    )
    _make_hset(
        v, "h002",
        config={"model": {"dim": 8}, "data": {"init_args": {"batch_size": 16}}, "seed_everything": 999},
        sessions=[{"status": "completed", "steps": 16}],
        last_session_at=datetime.now(timezone.utc).isoformat(),
    )
    rows = _collect_rows(model_dir, version_filter=None)
    by_slug = {r.slug: r for r in rows}
    assert by_slug["v001/h002"].notes == ""


# ── run_list (top-level integration) ───────────────────────────────────────

def test_run_list_missing_model_dir_prints_message(tmp_path, capsys):
    args = _args_for_list(tmp_path / "missing.py")
    code = run_list(args, runs_root=tmp_path)
    assert code == 0
    captured = capsys.readouterr()
    assert "No runs found" in captured.err


def test_run_list_renders_rows(tmp_path, capsys, monkeypatch):
    model_dir = _build_model_dir(tmp_path)
    # Pretend the model file is `iris.py` next to runs/.
    model_file = tmp_path / "iris.py"
    model_file.write_text("# placeholder\n")
    # Force non-TTY so we exercise the plain renderer.
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    args = _args_for_list(model_file)
    code = run_list(args, runs_root=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    assert "v001/h001" in out
    assert "v001/h002" in out
    assert "v002/h001" in out
    assert "val_ce" in out  # column header picked from session records


def test_run_list_limit(tmp_path, capsys, monkeypatch):
    model_dir = _build_model_dir(tmp_path)
    model_file = tmp_path / "iris.py"
    model_file.write_text("# placeholder\n")
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    args = _args_for_list(model_file, limit=1, sort_by="perf")
    code = run_list(args, runs_root=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    # Best val_ce is v001/h002 (0.340) — that should be the only data row.
    assert "v001/h002" in out
    assert "v002/h001" not in out
    assert "v001/h001" not in out


# ── _render_plain output shape ─────────────────────────────────────────────

def test_render_plain_header_and_rows():
    rows = [
        _row(slug="v001/h001", notes="baseline", sessions=1,
             last_at="just now", last_at_secs=1.0,
             val_label="val_ce", val_value=0.857, steps=8, status="done",
             sample_line=""),
    ]
    out = _render_plain(rows, val_label="val_ce", total_width=100)
    lines = out.splitlines()
    assert "Hset" in lines[0]
    assert "val_ce" in lines[0]
    assert "v001/h001" in lines[1]
    assert "baseline" in lines[1]
    assert "0.857" in lines[1]


# ── _compute_sample_budget ─────────────────────────────────────────────────

def test_compute_sample_budget_wide_terminal():
    """At 200-col, the fixed columns + overhead consume ~92 chars; Samples
    gets the rest. The exact number isn't load-bearing; what matters is
    that it's positive and not tiny."""
    b = _compute_sample_budget(total_width=200, val_label="val_ce")
    assert b > 80, f"expected wide-terminal budget to leave room for samples, got {b}"


def test_compute_sample_budget_narrow_terminal_floored():
    """On a 70-col terminal there's no room left after fixed columns;
    floor at _SAMPLES_MIN_WIDTH so Samples is still minimally usable."""
    b = _compute_sample_budget(total_width=70, val_label="val_ce")
    assert b == _SAMPLES_MIN_WIDTH


def test_compute_sample_budget_longer_val_label_shrinks_samples():
    """Wider val_label means less room for samples (same total_width)."""
    short = _compute_sample_budget(total_width=200, val_label="val")
    longer = _compute_sample_budget(total_width=200, val_label="val_loss_epoch")
    assert longer < short
    # Difference is exactly len("val_loss_epoch") - max(len("val"), 7)
    # = 14 - 7 = 7
    assert short - longer == 7


def test_compute_sample_budget_100_col_default_non_tty():
    """The non-TTY default total_width=100. Sample budget should be small
    but at least _SAMPLES_MIN_WIDTH so each row stays parseable."""
    b = _compute_sample_budget(total_width=100, val_label="val_ce")
    assert b >= _SAMPLES_MIN_WIDTH


# ── _render_rich doesn't collapse columns with long samples ───────────────

def test_render_rich_long_sample_does_not_collapse_columns(monkeypatch, capsys):
    """The bug we're fixing: a multi-hundred-char Samples cell used to
    cause Rich's last-resort reducer to zero out small columns. After the
    fix, every column header is visible in the output."""
    import io as io_mod
    from rich.console import Console
    # Force the Console() built inside _render_rich to render to our buffer
    # at a known width.
    buf = io_mod.StringIO()
    fake_console = Console(file=buf, force_terminal=True, width=160,
                           color_system=None)
    monkeypatch.setattr("mlops.list_cmd.Console", lambda *a, **kw: fake_console,
                        raising=False)
    # _render_rich does `from rich.console import Console` at the top of the
    # function, so monkeypatch the import target as well.
    import rich.console as rich_console_mod
    monkeypatch.setattr(rich_console_mod, "Console",
                        lambda *a, **kw: fake_console)

    rows = [
        _row(slug="v001/h001", notes="", sessions=1, last_at="1h ago",
             last_at_secs=1.0, val_label="val_ce", val_value=0.5,
             steps=1000, status="done",
             sample_line="x" * 500),
    ]
    _render_rich(rows, val_label="val_ce")
    out = buf.getvalue()

    # Every fixed-width header must appear (proxy for "didn't collapse").
    for header in ("Hset", "Notes", "Sess", "Last", "val_ce", "Steps",
                   "Status", "Samples"):
        assert header in out, f"missing column header {header!r} in:\n{out}"


def test_render_rich_does_not_emit_500_char_sample_in_output(monkeypatch):
    """Truncation to sample_budget actually happens — the full 500-char
    sample text doesn't survive into the rendered output."""
    import io as io_mod
    from rich.console import Console
    buf = io_mod.StringIO()
    fake_console = Console(file=buf, force_terminal=True, width=120,
                           color_system=None)
    import rich.console as rich_console_mod
    monkeypatch.setattr(rich_console_mod, "Console",
                        lambda *a, **kw: fake_console)

    big = "abcdefghij" * 50  # 500 chars of a recognizable pattern
    rows = [
        _row(slug="v001/h001", notes="", sessions=1, last_at="1h ago",
             last_at_secs=1.0, val_label="val_ce", val_value=0.5,
             steps=1000, status="done", sample_line=big),
    ]
    _render_rich(rows, val_label="val_ce")
    out = buf.getvalue()

    # The full 500-char string can't have survived in one piece.
    assert big not in out
