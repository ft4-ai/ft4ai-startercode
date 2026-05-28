"""Briefing.

Two renderings of the same hset-state briefing:

  * `render_briefing` — a styled Rich Panel for terminal display.
  * `render_briefing_plain` — a plain-text string for non-terminal output
    (`> log.txt`, `| tee`, etc.). No box characters, no ANSI escapes, no
    column-width assumptions.

Both are built once in `run.py` and handed to `BriefingPrintCallback`, which
prints one or the other at on_train_start depending on whether stdout is a
TTY.

Three modes drive title verb and border colour:
  * Fresh    → "ft4 train from scratch:"          · blue border
  * Resume   → "ft4 resume training:"             · green border
  * Backtrack→ "ft4 backtrack and resume training:" · magenta border
Title also carries `· sNNN` (the next session number) so the student sees
the run id from the briefing without scrolling.

When mode == BACKTRACK, a dedicated `backtrack:` row appears between
`data` and `paths` explaining which axis is older (step / hset / version,
or a combination) and the consequence ("training past this point will
overwrite last.ckpt").

Row order (top to bottom):
  model    → what am I training?
  version  → which model version?  · modified N d ago
  hset     → which hparams?  · diff vs prior sibling hset · description
  ckpt     → resuming from where?  size and metrics inline; symlink target
             on a continuation line
  plan     → what's the stopping target?  · total time projection
  pace     → how long per epoch
  data     → on what data?  · batch count
  paths    → where do outputs go?  (clickable links on modern terminals)
  samples  → latest generated samples preview (generative models only)
  tip      → one-line teaching nudge from the 4-tier selector
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlops.briefing_tips import TipContext, pick_tip
from mlops.hset_state import (
    Ft4Args,
    HsetInfo,
    Mode,
    VersionInfo,
    diff_runconfigs,
    diff_vs_prior_hset,
    latest_ckpt_step,
    latest_sibling_session_pace,
    parse_ckpt_step,
)


# Mode → (border colour, title verb). Single source of truth for the
# three-mode look. Backtrack borrows magenta from the previous "resume"
# styling — the eye-catching colour now flags "you're branching off",
# which is the genuinely unusual case worth highlighting.
_MODE_BORDER = {
    Mode.FRESH: "blue",
    Mode.RESUME: "green",
    Mode.BACKTRACK: "magenta",
}
_MODE_VERB = {
    Mode.FRESH: "train from scratch",
    Mode.RESUME: "resume training",
    Mode.BACKTRACK: "backtrack and resume training",
}


@dataclass(frozen=True)
class TrainerSummary:
    """Briefing-relevant slice of the Lightning Trainer.

    Extracted in run.py so the renderer/tips don't depend on Lightning.
    """
    max_epochs: int | None = None
    max_steps: int | None = None
    val_check_interval: float | int | None = None
    gradient_clip_val: float | None = None


# ── public Rich renderer ──────────────────────────────────────────────────

def render_briefing(
    model_file: Path,
    version: VersionInfo,
    hset: HsetInfo,
    sessions: list[dict],
    ckpt_path: Path | None,
    *,
    pl_module: Any = None,
    datamodule: Any = None,
    args: Ft4Args | None = None,
    trainer: TrainerSummary | None = None,
    config: dict | None = None,
    runs_root: Path = Path("runs"),
    now: datetime | None = None,
    mode: Mode | None = None,
    latest_version: VersionInfo | None = None,
    latest_hset: HsetInfo | None = None,
    latest_step: int | None = None,
    curr_runconfig: dict | None = None,
    prev_runconfig: dict | None = None,
    prev_session_id: str | None = None,
) -> Panel:
    """Build the Rich-styled briefing Panel.

    `now` is injectable for deterministic tests of the time-humanizer.
    """
    last = sessions[-1] if sessions else None
    tip_ctx = TipContext(
        args=args,
        version=version,
        hset=hset,
        n_sessions=len(sessions),
        last_session=last,
        pl_module=pl_module,
        datamodule=datamodule,
        trainer_summary=trainer,
        now=now,
    )
    return _render(
        model_file=model_file,
        version=version,
        hset=hset,
        n_sessions=len(sessions),
        last_session=last,
        ckpt_path=ckpt_path,
        pl_module=pl_module,
        datamodule=datamodule,
        args=args,
        trainer=trainer,
        config=config,
        runs_root=runs_root,
        now=now,
        tip=pick_tip(tip_ctx),
        mode=mode,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
        curr_runconfig=curr_runconfig,
        prev_runconfig=prev_runconfig,
        prev_session_id=prev_session_id,
        rdesc=args.rdesc if args is not None else None,
    )


# ── public plain-text renderer ────────────────────────────────────────────

# Two-space indent, "key:" padded to a fixed width so all values align.
# Widest key is "backtrack" (9) + colon = 10; pad to 11 for a single space gap.
_KEY_WIDTH = 11                         # width of "key:" + trailing space(s)
_VAL_INDENT = " " * (2 + _KEY_WIDTH)    # continuation lines start under values
_TREE_INDENT = _VAL_INDENT + "  "       # path-tree children indent two more


def render_briefing_plain(
    model_file: Path,
    version: VersionInfo,
    hset: HsetInfo,
    sessions: list[dict],
    ckpt_path: Path | None,
    *,
    pl_module: Any = None,
    datamodule: Any = None,
    args: Ft4Args | None = None,
    trainer: TrainerSummary | None = None,
    config: dict | None = None,
    runs_root: Path = Path("runs"),
    now: datetime | None = None,
    mode: Mode | None = None,
    latest_version: VersionInfo | None = None,
    latest_hset: HsetInfo | None = None,
    latest_step: int | None = None,
    curr_runconfig: dict | None = None,
    prev_runconfig: dict | None = None,
    prev_session_id: str | None = None,
) -> str:
    """Build the plain-text briefing for non-TTY output.

    Same data as `render_briefing`, formatted as `key: value` lines with no
    panel borders, no ANSI escapes, and no width assumptions.
    """
    last = sessions[-1] if sessions else None
    n_sessions = len(sessions)
    next_session_id = f"s{n_sessions + 1:03d}"
    effective_mode = mode if mode is not None else (
        Mode.RESUME if ckpt_path is not None else Mode.FRESH
    )
    lines: list[str] = []

    # title — verb varies by mode (fresh/resume/backtrack).
    verb = _MODE_VERB[effective_mode]
    lines.append(
        f"ft4 {verb}: {model_file.name} · {version.name} · {hset.name}"
        f" · {next_session_id}"
    )

    # model
    if pl_module is not None:
        cls = pl_module.__class__
        n = _param_count(pl_module)
        suffix = f"  {_human_count(n)} params" if n is not None else ""
        lines.append(_row("model", f"{cls.__module__}.{cls.__name__}{suffix}"))
    else:
        lines.append(_row("model", model_file.stem))

    # version
    suffix = _version_modified_suffix(version, now=now)
    v_text = f"{version.name}  {suffix}" if suffix else version.name
    lines.append(_row("version", v_text))
    if version.description:
        lines.append(f"{_VAL_INDENT}\"{version.description}\"")

    # hset
    if n_sessions == 0:
        hset_value = f"{hset.name}  NEW"
    else:
        plural = "" if n_sessions == 1 else "s"
        last_iso = last.get("ended_at") if last else None
        when_str = _humanize_session_age(last_iso, now=now)
        if when_str:
            hset_value = f"{hset.name}  {n_sessions} session{plural}, last {when_str}"
        else:
            hset_value = f"{hset.name}  {n_sessions} session{plural}"
    lines.append(_row("hset", hset_value))
    diff_line = _hset_diff_text(version, hset, config)
    if diff_line:
        lines.append(f"{_VAL_INDENT}{diff_line}")
    if hset.description:
        lines.append(f"{_VAL_INDENT}\"{hset.description}\"")

    # runconfig (training recipe)
    rc_lines = _runconfig_row_lines(
        curr_runconfig, prev_runconfig,
        rdesc=args.rdesc if args is not None else None,
        prev_session_id=prev_session_id,
    )
    if rc_lines:
        lines.append(_row("runconfig", rc_lines[0]))
        for extra in rc_lines[1:]:
            lines.append(f"{_VAL_INDENT}{extra}")

    # ckpt — size and metrics ride inline; symlink target on continuation line.
    lines.append(_row("ckpt", _ckpt_value_plain(ckpt_path, last)))
    # The symlink/step target line lives only in the paths tree below —
    # showing it under the ckpt row too would be redundant.

    # plan + pace: compute per-epoch estimate once; pace gets the rate,
    # plan gets the projected total time.
    per_epoch = _per_epoch_estimate(version, hset, last)
    sec_per_epoch = per_epoch[0] if per_epoch else None

    plan = _plan_text(trainer, last, sec_per_epoch=sec_per_epoch)
    if plan:
        lines.append(_row("plan", plan))

    pace = _pace_text(per_epoch)
    if pace:
        lines.append(_row("pace", pace))

    # data
    if datamodule is not None:
        lines.append(_row("data", _datamodule_summary(datamodule, config=config)))

    # backtrack row (only when mode == BACKTRACK)
    if effective_mode == Mode.BACKTRACK:
        bt_lines = _backtrack_row_lines(
            version=version, hset=hset, ckpt_path=ckpt_path,
            latest_version=latest_version, latest_hset=latest_hset,
            latest_step=latest_step,
        )
        bt_lines = [l for l in bt_lines if l]  # drop empty continuations
        if bt_lines:
            lines.append(_row("backtrack", bt_lines[0]))
            for extra in bt_lines[1:]:
                lines.append(f"{_VAL_INDENT}{extra}")

    # paths
    items = _path_items(hset, ckpt_path, last, pl_module=pl_module)
    lines.append(_row("paths", f"{_display_path(hset.dir)}/"))
    if items:
        # Pre-flatten so the alignment column width accounts for nested
        # children (which carry the same alignment as their siblings).
        flat: list[tuple[str, str]] = []
        for it in items:
            flat.append((it.name + it.suffix, it.annotation))
            for ch in it.children:
                flat.append(("  " + ch.name + ch.suffix, ch.annotation))
        label_w = max(len(label) for label, _ in flat)
        for label, annotation in flat:
            if annotation:
                lines.append(f"{_TREE_INDENT}{label:<{label_w + 2}}{annotation}")
            else:
                lines.append(f"{_TREE_INDENT}{label}")

    # samples — latest generated samples preview (generative models only).
    # Plain renderer has no styling, so prompt + continuation just run
    # together. Rich renderer dims the prompt to mark its boundary.
    sample_pairs = _samples_preview_pairs(hset)
    if sample_pairs:
        first = sample_pairs[0]
        lines.append(_row("samples", f"▸ {first[0]}{first[1]}"))
        for prompt, cont in sample_pairs[1:]:
            lines.append(f"{_VAL_INDENT}▸ {prompt}{cont}")

    # tip
    tip_ctx = TipContext(
        args=args,
        version=version,
        hset=hset,
        n_sessions=n_sessions,
        last_session=last,
        pl_module=pl_module,
        datamodule=datamodule,
        trainer_summary=trainer,
        now=now,
    )
    tip = pick_tip(tip_ctx)
    if tip:
        lines.append(_row("tip", tip))

    return "\n".join(lines)


def _row(key: str, value: str) -> str:
    return f"  {(key + ':'):<{_KEY_WIDTH}}{value}"


# ── shared Rich renderer ──────────────────────────────────────────────────

def _render(
    *,
    model_file: Path,
    version: VersionInfo,
    hset: HsetInfo,
    n_sessions: int,
    last_session: Optional[dict],
    ckpt_path: Path | None,
    pl_module: Any,
    datamodule: Any,
    args: Ft4Args | None,
    trainer: TrainerSummary | None,
    config: dict | None,
    runs_root: Path,
    now: datetime | None,
    tip: Optional[str],
    mode: Mode | None,
    latest_version: VersionInfo | None,
    latest_hset: HsetInfo | None,
    latest_step: int | None,
    curr_runconfig: dict | None,
    prev_runconfig: dict | None,
    prev_session_id: str | None,
    rdesc: str | None,
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")

    # model
    model_value = Text()
    if pl_module is not None:
        model_value.append(f"{pl_module.__class__.__module__}.{pl_module.__class__.__name__}")
        n_params = _param_count(pl_module)
        if n_params is not None:
            model_value.append(f"  {_human_count(n_params)} params", style="dim")
    else:
        model_value.append(model_file.stem, style="dim")
    table.add_row("model", model_value)

    # version
    v_value = Text()
    v_value.append(version.name, style="yellow")
    suffix = _version_modified_suffix(version, now=now)
    if suffix:
        v_value.append(f"  {suffix}", style="dim")
    table.add_row("version", v_value)
    if version.description:
        table.add_row("", Text(f'"{version.description}"', style="italic dim"))

    # hset
    h_status = Text()
    h_status.append(hset.name, style="yellow")
    if n_sessions == 0:
        h_status.append("  NEW", style="bold green")
    else:
        plural = "" if n_sessions == 1 else "s"
        last_iso = last_session.get("ended_at") if last_session else None
        when_str = _humanize_session_age(last_iso, now=now)
        if when_str:
            h_status.append(f"  {n_sessions} session{plural}, last {when_str}", style="dim")
        else:
            h_status.append(f"  {n_sessions} session{plural}", style="dim")
    table.add_row("hset", h_status)
    diff_line = _hset_diff_text(version, hset, config)
    if diff_line:
        table.add_row("", Text(diff_line, style="dim"))
    if hset.description:
        table.add_row("", Text(f'"{hset.description}"', style="italic dim"))

    # runconfig (training recipe: lr, optimizer, precision, …) — same-hset
    # diff against the prior session's recorded recipe makes "same model,
    # different recipe" runs visible at a glance.
    rc_lines = _runconfig_row_lines(
        curr_runconfig, prev_runconfig,
        rdesc=rdesc, prev_session_id=prev_session_id,
    )
    if rc_lines:
        table.add_row("runconfig", Text(rc_lines[0]))
        for extra in rc_lines[1:]:
            table.add_row("", Text(extra, style="dim"))

    # ckpt — size and inline metrics; symlink target on continuation line.
    table.add_row("ckpt", _ckpt_value(ckpt_path, last_session))
    # No continuation line for symlink/step target — paths tree owns that.

    # plan + pace: per-epoch estimate computed once and threaded into both
    # (plan carries the total-time projection; pace carries the rate).
    per_epoch = _per_epoch_estimate(version, hset, last_session)
    sec_per_epoch = per_epoch[0] if per_epoch else None

    plan = _plan_text(trainer, last_session, sec_per_epoch=sec_per_epoch)
    if plan:
        table.add_row("plan", Text(plan))

    pace = _pace_text(per_epoch)
    if pace:
        table.add_row("pace", Text(pace))

    # data
    if datamodule is not None:
        table.add_row("data", _datamodule_summary(datamodule, config=config))

    effective_mode = mode if mode is not None else (
        Mode.RESUME if ckpt_path is not None else Mode.FRESH
    )

    # backtrack row (mode == BACKTRACK only) — magenta to match the border.
    if effective_mode == Mode.BACKTRACK:
        bt_lines = _backtrack_row_lines(
            version=version, hset=hset, ckpt_path=ckpt_path,
            latest_version=latest_version, latest_hset=latest_hset,
            latest_step=latest_step,
        )
        bt_lines = [l for l in bt_lines if l]
        if bt_lines:
            table.add_row("backtrack", Text(bt_lines[0], style="magenta"))
            for extra in bt_lines[1:]:
                table.add_row("", Text(extra, style="magenta dim"))

    # paths
    table.add_row("", "")  # spacer
    table.add_row("paths", _path_tree(hset, ckpt_path, last_session, pl_module=pl_module))

    # samples preview (only when latest samples/step_*.md exists). The
    # prompt is rendered dim to mark its boundary; the continuation uses
    # the default style so the model's output reads naturally.
    sample_pairs = _samples_preview_pairs(hset)
    for i, (prompt, cont) in enumerate(sample_pairs):
        t = Text()
        t.append("▸ ", style="dim")
        t.append(prompt, style="dim")
        t.append(cont)
        table.add_row("samples" if i == 0 else "", t)

    # tip
    if tip:
        table.add_row("", "")
        table.add_row("tip", Text(tip, style="dim italic"))

    verb = _MODE_VERB[effective_mode]
    border_style = _MODE_BORDER[effective_mode]
    next_session_id = f"s{n_sessions + 1:03d}"

    title = Text()
    title.append(f"ft4 {verb}: ", style="bold")
    title.append(f"{model_file.name}", style="cyan")
    title.append(" · ")
    title.append(version.name, style="yellow")
    title.append(" · ")
    title.append(hset.name, style="yellow")
    title.append(" · ")
    title.append(next_session_id, style="yellow")

    return Panel(table, title=title, title_align="left", border_style=border_style)


# ── ckpt rendering ────────────────────────────────────────────────────────

def _ckpt_value(ckpt_path: Path | None, last_session: Optional[dict] = None) -> Text:
    """Rich `ckpt` row value: `<name>  <size>  metric=v  metric=v`.

    Size is omitted when the file is unreadable; metrics are omitted when
    last_session has none. Symlink target goes on a continuation line via
    the caller — not appended here.
    """
    if ckpt_path is None:
        return Text("none — will train from scratch", style="dim italic")
    t = Text()
    t.append(ckpt_path.name, style="bold")
    size = _ckpt_size_bytes(ckpt_path)
    if size is not None:
        t.append(f"  {_humanize_bytes(size)}", style="dim")
    if last_session is not None:
        for label, v in _metric_parts(last_session):
            t.append(f"  {label}={v:.3f}", style="dim")
    return t


def _ckpt_value_plain(ckpt_path: Path | None, last_session: Optional[dict] = None) -> str:
    if ckpt_path is None:
        return "none — will train from scratch"
    parts = [ckpt_path.name]
    size = _ckpt_size_bytes(ckpt_path)
    if size is not None:
        parts.append(_humanize_bytes(size))
    if last_session is not None:
        for label, v in _metric_parts(last_session):
            parts.append(f"{label}={v:.3f}")
    return "  ".join(parts)


def _resolve_symlink_target(path: Path) -> Optional[str]:
    """Return the symlink target's name if `path` is a symlink, else None."""
    try:
        if path.is_symlink():
            return Path(path.readlink()).name
    except (OSError, ValueError):
        pass
    return None


def _last_ckpt_target(ckpt_path: Path, hset_dir: Path) -> Optional[str]:
    """Resolve what `last.ckpt` points at — either as a real symlink target,
    or (since Lightning's `save_last=true` writes a file copy, not a symlink)
    by inferring from the highest-numbered `step=N.ckpt` in the same dir.

    Returns None for explicit step ckpts: their filename already carries
    the answer, so there's nothing to resolve.
    """
    target = _resolve_symlink_target(ckpt_path)
    if target is not None:
        return target
    if ckpt_path.name != "last.ckpt":
        return None
    step = latest_ckpt_step(hset_dir)
    return f"step={step}.ckpt" if step is not None else None


# ── metrics extraction (used by inline ckpt-row rendering) ────────────────

def _metric_parts(record: dict) -> list[tuple[str, float]]:
    """Shared metric extraction for Rich and plain renderers."""
    parts: list[tuple[str, float]] = []
    for prefix in ("train", "val"):
        key = record.get(f"{prefix}_metric_key")
        val = record.get(f"{prefix}_metric_value")
        if isinstance(key, str) and isinstance(val, (int, float)):
            parts.append((_display_metric_key(key), float(val)))
    return parts


def _display_metric_key(key: str) -> str:
    if key.endswith("_epoch"):
        return key[: -len("_epoch")]
    return key


def _humanize_bytes(n: int) -> str:
    """Coarse size formatter for the ckpt row. Integer-only — these sizes
    are *rounded*, not estimated, so no `≈` decoration at the call site.
    KB/MB/GB boundaries use 1024 to match `ls -h` and `du -h`."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n // 1024} KB"
    if n < 1024 ** 3:
        return f"{n // (1024 ** 2)} MB"
    return f"{n // (1024 ** 3)} GB"


def _ckpt_size_bytes(ckpt_path: Path | None) -> Optional[int]:
    """`stat()` size of the resolved symlink target (so we report disk usage,
    not the tiny symlink itself). None if the file is missing or unreadable."""
    if ckpt_path is None:
        return None
    try:
        return ckpt_path.stat().st_size  # follows symlinks
    except OSError:
        return None


def _humanize_seconds(secs: float) -> str:
    if secs < 1:
        return f"{secs * 1000:.0f} ms"
    if secs < 60:
        return f"{secs:.1f} sec"
    if secs < 3600:
        m = int(secs // 60)
        s = int(secs % 60)
        return f"{m} min {s:02d} sec"
    if secs < 86400:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        return f"{h} h {m:02d} min"
    d = int(secs // 86400)
    h = int((secs % 86400) // 3600)
    return f"{d} d {h:02d} h"


# ── path tree (shared data) ───────────────────────────────────────────────

@dataclass
class _PathItem:
    name: str           # display name, e.g. "checkpoints/last.ckpt"
    path: Path          # filesystem path (used by Rich renderer for file:// links)
    suffix: str         # optional inline suffix, e.g. " → step=12345.ckpt"
    annotation: str     # right-hand annotation, e.g. "params @ step 12345"
    bold: bool = False  # whether the Rich renderer should bold the name
    children: list["_PathItem"] = field(default_factory=list)


def _path_items(
    hset: HsetInfo,
    ckpt_path: Path | None,
    last_session: Optional[dict],
    pl_module: Any = None,
) -> list[_PathItem]:
    """Data describing the path-tree entries. Consumed by both renderers.

    Empty-state entries (e.g. `checkpoints/  empty — will fill during fit`)
    appear on fresh hsets so the tree shows where output will land.
    """
    out: list[_PathItem] = []
    is_generative = pl_module is not None and hasattr(pl_module, "generate_samples")

    cfg = hset.dir / "config.yaml"
    if cfg.exists():
        out.append(_PathItem("config.yaml", cfg, "", "full config"))

    last_ckpt = hset.dir / "checkpoints" / "last.ckpt"
    ckpt_dir = hset.dir / "checkpoints"
    if last_ckpt.exists():
        target = _last_ckpt_target(last_ckpt, hset.dir)
        suffix = f"  -> {target}" if target else ""
        # The session's global_step can be ahead of the last successful save
        # (e.g. session errored after the most recent ckpt was written). Read
        # the step from the file we actually have on disk.
        step = parse_ckpt_step(last_ckpt) or latest_ckpt_step(hset.dir)
        annotation = f"params @ step {step}" if step is not None else "params"
        out.append(_PathItem("checkpoints/last.ckpt", last_ckpt, suffix, annotation, bold=True))
    elif ckpt_dir.is_dir():
        out.append(_PathItem(
            "checkpoints/", ckpt_dir, "", "empty — will fill during fit",
        ))

    samples = hset.dir / "samples"
    sample_files = sorted(samples.glob("step_*.md")) if samples.is_dir() else []
    if sample_files:
        latest = sample_files[-1]
        out.append(_PathItem(f"samples/{latest.name}", latest, "", "latest output"))
    elif is_generative:
        out.append(_PathItem(
            "samples/", samples, "", "empty — will fill during fit",
        ))

    sessions_dir = hset.dir / "sessions"
    if sessions_dir.is_dir():
        session_subdirs = sorted(
            (p for p in sessions_dir.iterdir() if p.is_dir() and p.name.startswith("s")),
            key=lambda p: p.name,
        )
        if session_subdirs:
            latest = session_subdirs[-1]
            annotation = f"last session" if len(session_subdirs) == 1 else (
                f"last of {len(session_subdirs)}"
            )
            kids: list[_PathItem] = []
            hparams = latest / "hparams.yaml"
            if hparams.exists():
                kids.append(_PathItem("hparams.yaml", hparams, "", "hparams snapshot"))
            rc = latest / "runconfig.yaml"
            if rc.exists():
                kids.append(_PathItem("runconfig.yaml", rc, "", "runconfig used"))
            out.append(_PathItem(
                f"sessions/{latest.name}/", latest, "", annotation, children=kids,
            ))

    metrics_csv = hset.dir / "metrics.csv"
    if metrics_csv.exists():
        out.append(_PathItem("metrics.csv", metrics_csv, "", "cumulative"))

    return out


def _path_tree(
    hset: HsetInfo,
    ckpt_path: Path | None,
    last_session: Optional[dict],
    pl_module: Any = None,
) -> Text:
    """Rich-styled clickable path tree (consumes the shared `_path_items`).

    Children of a `_PathItem` render as a small sub-tree under their parent.
    Only used for `sessions/sNNN/` today, but generalized so future expansion
    targets (e.g. samples/) can re-use the same shape.
    """
    text = Text()
    text.append(f"{_display_path(hset.dir)}/\n", style="bold")

    items = _path_items(hset, ckpt_path, last_session, pl_module=pl_module)
    for i, item in enumerate(items):
        is_last_top = i == len(items) - 1
        _append_path_row(text, item, prefix="", is_last=is_last_top)
        for j, child in enumerate(item.children):
            child_prefix = "   " if is_last_top else "│  "
            _append_path_row(
                text, child, prefix=child_prefix, is_last=(j == len(item.children) - 1),
            )
        if not is_last_top or item.children:
            # Trailing newline handled by _append_path_row; remove the one
            # after the very last child of the very last top-level item.
            pass
    # Strip the final newline so the table cell doesn't end with a blank line.
    if text.plain.endswith("\n"):
        text = text[:-1]
    return text


def _append_path_row(text: Text, item: _PathItem, *, prefix: str, is_last: bool) -> None:
    """Append one tree row (with its connector + link + suffix + annotation
    + trailing newline) to `text`."""
    connector = "└─ " if is_last else "├─ "
    text.append(prefix, style="dim")
    text.append(connector, style="dim")
    link_style = "bold" if item.bold else ""
    text.append_text(_link(item.path, item.name, style=link_style))
    if item.suffix:
        text.append(item.suffix, style="dim")
    if item.annotation:
        text.append(f"  {item.annotation}", style="dim italic")
    text.append("\n")


def _link(path: Path, label: str, *, style: str = "") -> Text:
    """Return `label` as Rich Text with a `file://` hyperlink."""
    abs_path = path.resolve()
    link_style = f"link file://{abs_path}"
    return Text(label, style=f"{style} {link_style}".strip())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# ── runconfig row ─────────────────────────────────────────────────────────

# How many runconfig keys to surface inline in the summary line / diff
# continuation. Caps the row width on long configs.
_RUNCONFIG_SUMMARY_KEYS = 3
_RUNCONFIG_DIFF_KEYS = 4

# Keys that are honest runconfig members but uninteresting in the vs-prev
# diff display: the seed is auto-randomized every invocation (in
# main._build_lightning_cli's seed_everything_default), and max_epochs /
# max_steps bumps are how users add training, not recipe changes. These
# stay in `runconfig.yaml` for the audit trail; the renderer just hides
# them from the diff line.
_RUNCONFIG_DIFF_NOISE: frozenset[str] = frozenset({
    "seed_everything",
    "trainer.max_epochs",
    "trainer.max_steps",
    "trainer.max_time",
})


def _runconfig_row_lines(
    curr_runconfig: dict | None,
    prev_runconfig: dict | None,
    *,
    rdesc: str | None,
    prev_session_id: str | None,
) -> list[str]:
    """Build lines for the runconfig row.

    Line 1 (value):
      - quoted rdesc if set, e.g. `"warmup + cosine, lr=3e-4"`.
      - else a one-line summary like `lr=3e-4, weight_decay=0.01, precision=bf16-mixed`.
      - else empty list (no row).
    Line 2+ (continuation, only when prev_runconfig differs from curr):
      `vs sPREV: lr 1e-4->3e-4, precision fp32->bf16-mixed`.
    """
    lines: list[str] = []

    if rdesc:
        lines.append(f'"{rdesc}"')
    elif curr_runconfig:
        summary = _runconfig_summary(curr_runconfig)
        if summary:
            lines.append(summary)
    else:
        # Truly nothing to show — caller should skip the row.
        return []

    if prev_runconfig is not None:
        diffs = [
            d for d in diff_runconfigs(curr_runconfig, prev_runconfig)
            if d[0] not in _RUNCONFIG_DIFF_NOISE
        ]
        if diffs:
            label = f"vs {prev_session_id}" if prev_session_id else "vs prev"
            head = diffs[:_RUNCONFIG_DIFF_KEYS]
            parts = [
                f"{_clean_runconfig_key(k)} {_fmt_diff_val(old)}->{_fmt_diff_val(new)}"
                for k, old, new in head
            ]
            extra = ""
            if len(diffs) > _RUNCONFIG_DIFF_KEYS:
                extra = f" +{len(diffs) - _RUNCONFIG_DIFF_KEYS} more"
            lines.append(f"{label}: " + ", ".join(parts) + extra)

    return lines


def _runconfig_summary(runconfig: dict) -> str:
    """Compact one-liner for runconfig when no rdesc is set. Picks the
    most-recognizable keys (lr, weight_decay, precision, optimizer, …) up to
    `_RUNCONFIG_SUMMARY_KEYS`."""
    priority = (
        "model.lr", "model.init_args.lr",
        "model.learning_rate", "model.init_args.learning_rate",
        "trainer.precision",
        "model.optimizer", "model.init_args.optimizer",
        "model.weight_decay", "model.init_args.weight_decay",
        "model.warmup_steps", "model.init_args.warmup_steps",
        "model.dropout", "model.init_args.dropout",
        "data.batch_size", "data.init_args.batch_size",
        "seed_everything",
    )
    parts: list[str] = []
    seen: set[str] = set()
    for key in priority:
        if key in runconfig and runconfig[key] is not None:
            label = _clean_runconfig_key(key)
            if label in seen:
                continue
            seen.add(label)
            parts.append(f"{label}={_fmt_diff_val(runconfig[key])}")
            if len(parts) >= _RUNCONFIG_SUMMARY_KEYS:
                break
    return ", ".join(parts)


def _clean_runconfig_key(key: str) -> str:
    """Strip jsonargparse's `.init_args.` boilerplate so keys read as
    `model.lr` rather than `model.init_args.lr` in user-facing copy."""
    return key.replace(".init_args.", ".")


# ── backtrack row ─────────────────────────────────────────────────────────

def _backtrack_row_lines(
    *,
    version: VersionInfo,
    hset: HsetInfo,
    ckpt_path: Path | None,
    latest_version: VersionInfo | None,
    latest_hset: HsetInfo | None,
    latest_step: int | None,
) -> list[str]:
    """Build the lines of the dedicated backtrack row.

    Content varies by which axis is older (version, hset, step, or a combo).
    First line is the headline; any further lines are continuations rendered
    under the value column. Returns [] for non-backtrack modes (caller should
    only invoke this when mode == BACKTRACK).
    """
    version_back = (
        latest_version is not None and version.name != latest_version.name
    )
    hset_back = (
        latest_hset is not None and hset.name != latest_hset.name
    )
    loaded_step = parse_ckpt_step(ckpt_path)
    step_back = (
        loaded_step is not None and latest_step is not None
        and loaded_step != latest_step
    )

    # Combined backtrack (more than one axis) — one consolidated headline
    # so we don't repeat ourselves across multiple rows.
    n_axes = sum([version_back, hset_back, step_back])
    if n_axes >= 2:
        head_pieces = [f"resumed {version.name}/{hset.name}"]
        if loaded_step is not None:
            head_pieces.append(f"@ step {loaded_step}")
        head = " ".join(head_pieces)
        latest_pieces = []
        if latest_version is not None:
            latest_pieces.append(latest_version.name)
        if latest_hset is not None:
            latest_pieces.append(latest_hset.name)
        latest_str = "/".join(latest_pieces) if latest_pieces else ""
        if latest_step is not None:
            latest_str = (latest_str + " " if latest_str else "") + f"@ step {latest_step}"
        return [
            head,
            f"latest: {latest_str}" if latest_str else "",
            "this branch continues independently",
        ]

    if version_back:
        latest_name = latest_version.name if latest_version else "?"
        return [
            f"resumed {version.name} — latest version is {latest_name}",
            f"{version.name} continues independently; later versions are untouched",
        ]
    if hset_back:
        latest_name = latest_hset.name if latest_hset else "?"
        return [
            f"resumed {hset.name} — latest hset in {version.name} is {latest_name}",
            f"{hset.name} continues independently; {latest_name} is untouched",
        ]
    if step_back:
        loaded = ckpt_path.name if ckpt_path else "?"
        return [
            f"loaded {loaded} — latest is step={latest_step}",
            "training past this point will overwrite last.ckpt; "
            "use --new-hset to keep both branches",
        ]
    return []


# ── samples preview ───────────────────────────────────────────────────────

# How many prompt/continuation pairs to show, and the per-line char budget.
# Fixed budget — assumes the terminal is wide enough; threading TTY width
# through every renderer adds complexity for no real payoff (samples are
# easily inspectable in the linked file if a line gets truncated).
_SAMPLES_MAX_PAIRS = 3
_SAMPLES_LINE_CHARS = 85


def _samples_preview_pairs(hset: HsetInfo) -> list[tuple[str, str]]:
    """Return up to `_SAMPLES_MAX_PAIRS` (prompt, continuation) pairs from
    the latest `<hset>/samples/step_*.md`, or [].

    The pair is already truncated together to fit `_SAMPLES_LINE_CHARS`
    (accounting for the bullet prefix), so renderers can format the prompt
    and continuation distinctly (e.g. dim the prompt in Rich, leave the
    continuation default) without re-doing the math.

    Lazy import: live_sample_callback pulls in Lightning at module load,
    so we defer that cost to the point of use.
    """
    samples_dir = hset.dir / "samples"
    if not samples_dir.is_dir():
        return []
    files = sorted(samples_dir.glob("step_*.md"))
    if not files:
        return []
    try:
        text = files[-1].read_text()
    except OSError:
        return []
    from mlops.live_sample_callback import extract_pairs
    pairs = extract_pairs(text)[:_SAMPLES_MAX_PAIRS]
    bullet = "▸ "
    out: list[tuple[str, str]] = []
    for prompt, continuation in pairs:
        # Truncate against the rendered line length (bullet + prompt +
        # continuation). When the prompt itself is long enough to overflow,
        # truncate inside it; otherwise truncate the continuation.
        budget = _SAMPLES_LINE_CHARS - len(bullet)
        if len(prompt) > budget:
            prompt = prompt[: budget - 1] + "…"
            continuation = ""
        elif len(prompt) + len(continuation) > budget:
            continuation = continuation[: budget - len(prompt) - 1] + "…"
        out.append((prompt, continuation))
    return out


# ── helpers ───────────────────────────────────────────────────────────────

def _param_count(pl_module: Any) -> Optional[int]:
    try:
        return sum(p.numel() for p in pl_module.parameters())
    except Exception:
        return None


def _human_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f} K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f} M"
    return f"{n / 1_000_000_000:.2f} B"


def _datamodule_summary(datamodule: Any, *, config: dict | None = None) -> str:
    """Render the `data` row value.

    Goal shape: `<Class>(arg=v, arg=v)  N records  ·  M batches`. Init-arg
    summary surfaces *which* dataset/slice (e.g. `data_size=full`) — the
    info the bare class name leaves invisible. Batches require batch_size,
    pulled in this order: datamodule.batch_size attr →
    config.data[.init_args].batch_size → omit. Sample-count failures fall
    back to just the class-plus-args head.

    Caveat: most real datamodules populate `train_dataset` inside `setup()`,
    which runs INSIDE `trainer.fit` — i.e. after the briefing has rendered.
    For those, we'll only ever show the class name pre-fit. Auto-calling
    `setup()` here would force eager dataset/tokenizer loading on every
    invocation (including read-only `ft4 show`), so we deliberately don't.
    """
    name = datamodule.__class__.__name__
    args = _datamodule_args_summary(config)
    head = f"{name}({args})" if args else name

    size: Optional[int] = None
    for attr in ("train_dataset", "dataset"):
        ds = getattr(datamodule, attr, None)
        if ds is not None and hasattr(ds, "__len__"):
            try:
                size = len(ds)
                break
            except (TypeError, AttributeError):
                pass
    if size is None:
        return head

    bs = _resolve_batch_size(datamodule, config)
    if bs is None or bs <= 0:
        return f"{head}  {size:,} records"
    # ceil division
    batches = (size + bs - 1) // bs
    return f"{head}  {size:,} records  ·  {batches:,} batches"


# Scientifically-meaningful data init_args, in display order. Excludes the
# data-loader knobs (batch_size, num_workers, …) — those are infrastructure,
# already covered by the "M batches" tail or scientifically uninteresting.
_DATA_ARG_KEYS: tuple[str, ...] = ("data_size", "name", "split", "path", "seq_len")
_DATA_ARG_LIMIT = 3


def _datamodule_args_summary(config: dict | None) -> str:
    """`key=v, key=v` from config.data[.init_args], picked from
    `_DATA_ARG_KEYS` and capped at `_DATA_ARG_LIMIT`. Empty string when
    config is missing or carries none of these keys."""
    if not isinstance(config, dict):
        return ""
    data = config.get("data") or {}
    if not isinstance(data, dict):
        return ""
    init = data.get("init_args")
    sources: list[dict] = []
    if isinstance(init, dict):
        sources.append(init)
    sources.append(data)
    parts: list[str] = []
    for key in _DATA_ARG_KEYS:
        for src in sources:
            if key in src and src[key] is not None:
                parts.append(f"{key}={_fmt_diff_val(src[key])}")
                break
        if len(parts) >= _DATA_ARG_LIMIT:
            break
    return ", ".join(parts)


def _resolve_batch_size(datamodule: Any, config: dict | None) -> Optional[int]:
    """Find batch_size: datamodule attribute first (fastest, most reliable),
    then fall back to the resolved config under data[.init_args].batch_size."""
    bs = getattr(datamodule, "batch_size", None)
    if isinstance(bs, int):
        return bs
    if not isinstance(config, dict):
        return None
    data = config.get("data") or {}
    if not isinstance(data, dict):
        return None
    init = data.get("init_args") or {}
    for src in (init if isinstance(init, dict) else {}, data):
        v = src.get("batch_size")
        if isinstance(v, int):
            return v
    return None


def _pace_text(per_epoch: tuple[float, str] | None) -> str:
    """The 'pace' row: per-epoch only.

    Returns 'measured after epoch 1' when we have no estimate yet, else
    '≈ Th Mmin/epoch  (source)'. Total-time projection moved to `_plan_text`.
    """
    if per_epoch is None:
        return "measured after epoch 1"
    secs, source = per_epoch
    return f"≈ {_humanize_seconds(secs)}/epoch  {source}"


def _per_epoch_estimate(
    version: VersionInfo, hset: HsetInfo, last_session: Optional[dict],
) -> tuple[float, str] | None:
    """Returns (sec_per_epoch, source-label) or None if no data."""
    if last_session is not None:
        est = last_session.get("est_sec_per_epoch")
        if isinstance(est, (int, float)) and est > 0:
            return float(est), "(recent)"
    sibling = latest_sibling_session_pace(version, hset)
    if sibling is not None:
        return sibling[1], f"(based on {sibling[0]})"
    return None


def _hset_diff_text(
    version: VersionInfo, hset: HsetInfo, config: dict | None,
) -> str:
    """Render the diff continuation line for the hset row.

    Returns 'vs hNNN: model.lr 1e-4→3e-4, ...' or empty string when:
      * no `config` provided
      * no prior sibling hset (this is the first hset in the version)
      * sibling's config.yaml missing/unparseable
      * configs differ in nothing scientifically meaningful
    Caps the displayed diffs at 5; appends '+K more' for the rest.
    """
    if config is None:
        return ""
    result = diff_vs_prior_hset(version, hset, config)
    if result is None:
        return ""
    sibling_name, diffs = result
    if not diffs:
        return ""
    MAX = 5
    head = diffs[:MAX]
    parts = [f"{k} {_fmt_diff_val(old)}->{_fmt_diff_val(new)}" for k, old, new in head]
    text = f"vs {sibling_name}: " + ", ".join(parts)
    if len(diffs) > MAX:
        text += f" +{len(diffs) - MAX} more"
    return text


def _fmt_diff_val(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    if isinstance(v, list):
        # Avoid blowing out width with a giant list repr.
        if len(v) > 3:
            return f"[{len(v)} items]"
        return repr([_fmt_diff_val(x) for x in v])
    s = str(v)
    # Path-like (anything containing a path separator) — show just the leaf.
    if "/" in s and len(s) > 30:
        return ".../" + s.rsplit("/", 1)[-1]
    if len(s) > 30:
        return s[:29] + "…"
    return repr(s) if isinstance(v, str) else str(v)


def _plan_text(
    trainer: Optional["TrainerSummary"],
    last_session: Optional[dict],
    *,
    sec_per_epoch: float | None = None,
) -> str:
    """The stopping target row, with an optional projected-total-time suffix.

    Fresh: 'train N epochs' / 'train up to N steps'.
    Resume: 'add ΔN epochs (total → N)' (epoch-based).
    Resume with missing epochs field: 'add up to N epochs (total → N)'.
    Step-based resume: 'add up to N steps' — no Δ.

    When `sec_per_epoch` is known and max_epochs is set, appends
    ' ≈ Th Mmin' covering the remaining work (delta) or the whole run.
    """
    if trainer is None:
        return ""
    max_epochs = trainer.max_epochs
    max_steps = trainer.max_steps
    is_resume = last_session is not None

    if isinstance(max_epochs, int) and max_epochs > 0:
        last_epochs = last_session.get("epochs") if is_resume else None
        if isinstance(last_epochs, int) and 0 < last_epochs < max_epochs:
            delta = max_epochs - last_epochs
            body = f"add {delta} epoch{'' if delta == 1 else 's'} (total -> {max_epochs})"
            remaining_epochs = delta
        elif is_resume:
            body = f"add up to {max_epochs} epoch{'' if max_epochs == 1 else 's'} (total -> {max_epochs})"
            remaining_epochs = max_epochs
        else:
            body = f"train {max_epochs} epoch{'' if max_epochs == 1 else 's'}"
            remaining_epochs = max_epochs
        if isinstance(sec_per_epoch, (int, float)) and sec_per_epoch > 0:
            total_secs = remaining_epochs * sec_per_epoch
            body += f"  ≈ {_humanize_seconds(total_secs)}"
        return body

    if isinstance(max_steps, int) and max_steps > 0:
        verb = "add up to" if is_resume else "train up to"
        return f"{verb} {max_steps:,} steps"

    return ""


def _version_modified_suffix(version: VersionInfo, *, now: datetime | None) -> str:
    """Returns 'modified N d ago' (or similar) for an existing version,
    empty string when the version was created within the last 60 seconds
    (i.e. this run is creating it; 'modified just now' would be noise).
    Also empty when created_at is missing (older runs lacked the field).
    """
    if not version.created_at:
        return ""
    try:
        when = datetime.fromisoformat(version.created_at)
    except (ValueError, TypeError):
        return ""
    age = _humanize_age(when, now=now)
    if age == "just now":
        return ""
    return f"modified {age}"


def _humanize_session_age(iso: Optional[str], *, now: datetime | None) -> Optional[str]:
    if not isinstance(iso, str):
        return None
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return _humanize_age(when, now=now)


def _humanize_age(when: datetime, *, now: datetime | None = None) -> str:
    """Coarse age formatter. Negative deltas (future) treated as 'just now'."""
    if now is None:
        now = datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    secs = max(0.0, (now - when).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"
