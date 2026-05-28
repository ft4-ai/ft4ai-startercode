"""ft4 list — tabular summary of versions × hsets for one model.

Read-only: walks `runs/<model_stem>/v*/h*/`, never mutates disk. No Lightning
imports — pure filesystem reads, so this command starts instantly.

The flat-row layout (one row per hset, with the version slug in the first
column) is chosen over grouped-with-banners because `--by-perf` ranking
across versions needs to read top-to-bottom uninterrupted.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlops.hset_state import (
    Ft4Args,
    HsetInfo,
    VersionInfo,
    all_hsets,
    all_versions,
    diff_hset_configs,
    read_sessions,
)
from mlops.live_sample_callback import extract_pairs


# Total table width when stdout isn't a TTY or terminal size can't be detected.
_FALLBACK_TOTAL_WIDTH = 100

# Fixed widths for every column except Samples (whose width is computed at
# render time from the available terminal width). Numbers are chosen as
# max(header_text, expected_content). For the val_label column we take
# max(len(val_label), 7) at render time since the label is dynamic.
#
# Why fixed widths and not min_width? Rich's _collapse_widths in rich/table.py
# treats a column as wrappable only when `column.width is None and not
# column.no_wrap`. With all our columns no_wrap, none are wrappable; when
# total wanted width exceeds the terminal, Rich's last-resort reducer
# (table.py:563-567) ignores min_width and zeros out small columns. Fixed
# widths take that codepath off the table entirely.
_FIXED_COL_WIDTHS = {
    "slug": 11,      # "v100/h100" = 9 chars; bump for 4-digit version/hset
    "notes": 22,     # arbitrary; bigger = more diff visible, less for samples
    "sessions": 5,   # header "Sess" = 4, content up to "999" = 3
    "last_at": 8,    # "99d ago" or "just now" = 8 chars
    # val column dynamic: max(len(val_label), 7)
    "steps": 7,      # "999k", "99.9M" fit in 6; pad for safety
    "status": 7,    # "intrpt", "errored" fit in 7
}
# Rich Table(pad_edge=False) uses (0,1) padding (1 space each side of cell)
# and 1 separator char per gap. For N columns: 2*N padding + (N+1) borders.
# Our 8 columns -> 25 chars overhead.
_TABLE_OVERHEAD = 3 * 8 + 1
# Floor for the Samples column so it stays usable on narrow terminals.
_SAMPLES_MIN_WIDTH = 20


def _val_col_width(val_label: str) -> int:
    """Width for the dynamic val_label column: header text, with a sane floor."""
    return max(len(val_label), 7)


def _compute_sample_budget(total_width: int, val_label: str) -> int:
    """Per-row character budget for the Samples cell.

    Caller pre-truncates sample_line to this many chars before handing it to
    Rich; combined with `width=N` (not min_width) on every fixed column, Rich
    never needs its last-resort reducer and small columns can't collapse.
    """
    fixed = sum(_FIXED_COL_WIDTHS.values()) + _val_col_width(val_label)
    budget = total_width - fixed - _TABLE_OVERHEAD
    return max(_SAMPLES_MIN_WIDTH, budget)

_SAMPLE_STEP_RE = re.compile(r"step_(\d+)\.md$")


@dataclass(frozen=True)
class Row:
    """One row per hset in the list output. All fields are display-ready
    strings except `val_value` (for sorting) and the numeric `sessions` /
    `steps` (also for sorting / formatting choice)."""
    slug: str               # "v001/h003"
    notes: str              # hdesc or hset-diff or ""
    sessions: int
    last_at: str            # "2d ago" or "—"
    last_at_secs: float     # raw Unix timestamp for sort; 0.0 if missing
    val_label: str          # e.g. "val_ce"; "" if no val metric anywhere
    val_value: float | None # min across this hset's sessions, for sorting
    steps: int
    status: str             # "done" / "intrpt" / "error" / "—"
    sample_line: str        # collapsed first prompt+completion, untruncated


# ── Entry point ───────────────────────────────────────────────────────────

def run_list(args: Ft4Args, *, runs_root: Path = Path("runs")) -> int:
    """Walk `runs/<stem>/`, build rows, sort, render."""
    model_dir = runs_root / args.model_file.stem
    if not model_dir.exists():
        print(
            f"No runs found for {args.model_file}. "
            f"Try `ft4 train {args.model_file}`.",
            file=sys.stderr,
        )
        return 0

    rows = _collect_rows(model_dir, version_filter=args.version)
    if not rows:
        print(f"No hsets found under {model_dir}.", file=sys.stderr)
        return 0

    rows = _sort_rows(rows, args.sort_by)
    if args.limit is not None:
        rows = rows[: args.limit]

    val_label = _pick_val_label(rows)

    if sys.stdout.isatty():
        _render_rich(rows, val_label)
    else:
        sys.stdout.write(_render_plain(rows, val_label, _FALLBACK_TOTAL_WIDTH))
        if rows:
            sys.stdout.write("\n")
    return 0


# ── Row collection ────────────────────────────────────────────────────────

def _collect_rows(model_dir: Path, *, version_filter: str | None) -> list[Row]:
    """For each (version, hset) pair under model_dir, build one Row. Versions
    without hsets are skipped (a version with no hsets means a fresh dir
    abandoned before phase 2 — the user can see this with `ls runs/`)."""
    rows: list[Row] = []
    now = datetime.now(timezone.utc)
    for version in all_versions(model_dir, newest_first=True):
        if version_filter is not None and version.name != version_filter:
            continue
        hsets = all_hsets(version.dir, newest_first=False)  # h001, h002, ... order
        baseline = hsets[0] if hsets else None
        for hset in hsets:
            rows.append(_collect_row(version, hset, baseline, now))
    return rows


def _collect_row(
    version: VersionInfo,
    hset: HsetInfo,
    baseline: HsetInfo | None,
    now: datetime,
) -> Row:
    sessions = read_sessions(hset.dir)
    notes = _notes_for(hset, baseline)
    last_at, last_at_secs = _last_session(hset.dir, now)
    val_label, val_value = _best_val(sessions)
    steps = sum(int(s.get("steps", 0) or 0) for s in sessions)
    status = _last_status(sessions)
    sample_line = _latest_sample_line(hset.dir)
    return Row(
        slug=f"{version.name}/{hset.name}",
        notes=notes,
        sessions=len(sessions),
        last_at=last_at,
        last_at_secs=last_at_secs,
        val_label=val_label,
        val_value=val_value,
        steps=steps,
        status=status,
        sample_line=sample_line,
    )


def _notes_for(hset: HsetInfo, baseline: HsetInfo | None) -> str:
    """`hdesc` if set; else config diff vs the version's first hset; else ""."""
    if hset.description:
        return hset.description
    if baseline is None or baseline.name == hset.name:
        return ""
    return diff_hset_configs(hset.dir, baseline.dir)


def _last_session(hset_dir: Path, now: datetime) -> tuple[str, float]:
    """Return (display_string, unix_timestamp). The display string is for the
    Last column; the timestamp is for stable time-based sorting."""
    path = hset_dir / "hset.json"
    if not path.exists():
        return "—", 0.0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "—", 0.0
    iso = data.get("last_session_at")
    if not iso:
        return "—", 0.0
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return "—", 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _relative_time(iso, now), when.timestamp()


def _relative_time(iso_str: str, now: datetime) -> str:
    """Format an ISO timestamp as 'just now' / '5m ago' / '3h ago' / '2d ago'.
    Returns '—' if the timestamp can't be parsed."""
    try:
        when = datetime.fromisoformat(iso_str)
    except ValueError:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = (now - when).total_seconds()
    if delta < 0:
        return "just now"
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    days = int(delta // 86400)
    return f"{days}d ago"


def _best_val(sessions: list[dict]) -> tuple[str, float | None]:
    """Pick the most common val_metric_key across sessions, return its
    minimum value. Returns ('', None) if no session logged a val metric."""
    label = ""
    best: float | None = None
    for s in sessions:
        v = s.get("val_metric_value")
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        k = s.get("val_metric_key") or ""
        if not label:
            label = k
        if best is None or vf < best:
            best = vf
    return label, best


def _last_status(sessions: list[dict]) -> str:
    if not sessions:
        return "—"
    raw = sessions[-1].get("status") or "—"
    # Compact for the column. Lightning's status values are short strings.
    return {"completed": "done", "interrupted": "intrpt"}.get(raw, raw)


def _latest_sample_line(hset_dir: Path) -> str:
    """Read the newest `samples/step_NNNNNN.md` and return its first
    `prompt+completion` pair as a single line. Empty string if no samples."""
    samples_dir = hset_dir / "samples"
    if not samples_dir.exists():
        return ""
    best_step = -1
    best_path: Path | None = None
    for p in samples_dir.iterdir():
        m = _SAMPLE_STEP_RE.search(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p
    if best_path is None:
        return ""
    try:
        text = best_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    pairs = extract_pairs(text)
    if not pairs:
        return ""
    prompt, continuation = pairs[0]
    # No upstream truncation here — both renderers truncate per their own
    # computed budget (see _compute_sample_budget). Newlines and CR are
    # collapsed to single spaces so the sample fits on one row.
    return (prompt + continuation).replace("\n", " ").replace("\r", " ")


# ── Sorting ──────────────────────────────────────────────────────────────

def _sort_rows(rows: list[Row], sort_by: str) -> list[Row]:
    if sort_by == "perf":
        # Best val first; rows without a val metric sink to the bottom.
        return sorted(
            rows,
            key=lambda r: (r.val_value is None, r.val_value if r.val_value is not None else 0.0),
        )
    # Default: by recency. Rows without a recorded last_session sink to the
    # bottom (last_at_secs == 0.0). Negate the timestamp so newest comes
    # first under an ascending sort.
    return sorted(
        rows,
        key=lambda r: (r.last_at_secs == 0.0, -r.last_at_secs),
    )


def _pick_val_label(rows: list[Row]) -> str:
    """Pick the most common val_label among rows for the column header.
    Returns 'val' as a fallback when no row has a labeled val metric."""
    counts: dict[str, int] = {}
    for r in rows:
        if r.val_label:
            counts[r.val_label] = counts.get(r.val_label, 0) + 1
    if not counts:
        return "val"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ── Rendering ────────────────────────────────────────────────────────────

def _format_steps(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _format_val(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _render_rich(rows: list[Row], val_label: str) -> None:
    """Render via Rich with fixed widths everywhere. The Samples column is
    sized from the terminal width and rows are pre-truncated so Rich never
    needs its last-resort reducer (which would ignore min_width and zero out
    small columns)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    total_width = console.width or _FALLBACK_TOTAL_WIDTH
    sample_budget = _compute_sample_budget(total_width, val_label)
    notes_width = _FIXED_COL_WIDTHS["notes"]

    table = Table(show_lines=False, pad_edge=False)
    table.add_column("Hset", no_wrap=True, width=_FIXED_COL_WIDTHS["slug"])
    table.add_column("Notes", no_wrap=True, width=notes_width, overflow="ellipsis")
    table.add_column("Sess", no_wrap=True, justify="right", width=_FIXED_COL_WIDTHS["sessions"])
    table.add_column("Last", no_wrap=True, width=_FIXED_COL_WIDTHS["last_at"])
    table.add_column(val_label, no_wrap=True, justify="right", width=_val_col_width(val_label))
    table.add_column("Steps", no_wrap=True, justify="right", width=_FIXED_COL_WIDTHS["steps"])
    table.add_column("Status", no_wrap=True, width=_FIXED_COL_WIDTHS["status"])
    table.add_column("Samples", no_wrap=True, width=sample_budget, overflow="ellipsis")

    for r in rows:
        table.add_row(
            r.slug,
            _truncate(r.notes, notes_width),
            str(r.sessions),
            r.last_at,
            _format_val(r.val_value),
            _format_steps(r.steps),
            r.status,
            _truncate(r.sample_line, sample_budget),
        )
    console.print(table)


def _render_plain(rows: list[Row], val_label: str, total_width: int) -> str:
    """Plain text table for non-TTY output. Fixed widths everywhere except
    Samples, which gets the remainder via the same _compute_sample_budget
    helper used by the Rich path so both renderers stay in lockstep."""
    sample_width = _compute_sample_budget(total_width, val_label)
    val_width = _val_col_width(val_label)

    header_cells = [
        _pad("Hset", _FIXED_COL_WIDTHS["slug"]),
        _pad("Notes", _FIXED_COL_WIDTHS["notes"]),
        _rpad("Sess", _FIXED_COL_WIDTHS["sessions"]),
        _pad("Last", _FIXED_COL_WIDTHS["last_at"]),
        _rpad(val_label, val_width),
        _rpad("Steps", _FIXED_COL_WIDTHS["steps"]),
        _pad("Status", _FIXED_COL_WIDTHS["status"]),
        "Samples",
    ]
    lines = [" ".join(header_cells).rstrip()]

    for r in rows:
        cells = [
            _pad(r.slug, _FIXED_COL_WIDTHS["slug"]),
            _pad(_truncate(r.notes, _FIXED_COL_WIDTHS["notes"]), _FIXED_COL_WIDTHS["notes"]),
            _rpad(str(r.sessions), _FIXED_COL_WIDTHS["sessions"]),
            _pad(r.last_at, _FIXED_COL_WIDTHS["last_at"]),
            _rpad(_format_val(r.val_value), val_width),
            _rpad(_format_steps(r.steps), _FIXED_COL_WIDTHS["steps"]),
            _pad(r.status, _FIXED_COL_WIDTHS["status"]),
            _truncate(r.sample_line, sample_width),
        ]
        lines.append(" ".join(cells).rstrip())
    return "\n".join(lines)


def _pad(s: str, w: int) -> str:
    """Left-align in a fixed width, truncating with ellipsis if too long."""
    s = _truncate(s, w)
    return s.ljust(w)


def _rpad(s: str, w: int) -> str:
    s = _truncate(s, w)
    return s.rjust(w)


def _truncate(s: str, w: int) -> str:
    if len(s) <= w:
        return s
    if w <= 1:
        return "…"
    return s[: w - 1] + "…"
