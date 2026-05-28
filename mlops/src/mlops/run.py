"""Smart flow for `ft4 train <model.py>`.

End-to-end orchestration:

  1. Discover the model class from the file.
  2. Build LightningCLI (instantiates model + datamodule from the YAML cascade).
  3. Compute the three signals (state_dict_hash, model_file_hash, config_hash).
  4. Resolve version → hset → checkpoint (read-only at this point).
  5. Build the briefing panel and hand it to BriefingPrintCallback — it'll
     print at on_train_start (inside Live), so the briefing lands above the
     samples and progress bar.
  6. Persist config and inject prompts onto sample callback.
  7. Phase-1 baseline samples (before creating session_dir, so a failed
     baseline doesn't leave an empty session_dir behind).
  8. Create session_dir, inject paths, start finalization callback.
  9. Run trainer.fit(). Lightning's exception-handling + the
     SessionFinalizationCallback take care of bookkeeping on any exit
     path (clean, Ctrl-C, error).

The session record (sessions.jsonl) and hset-level cumulative metrics
are written by SessionFinalizationCallback via Lightning's on_fit_end /
on_exception / teardown hooks — not by user code here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from mlops.briefing_display import (
    _humanize_seconds,
    render_briefing,
    render_briefing_plain,
)
from mlops.briefing_print_callback import BriefingPrintCallback
from mlops.session_finalization import SessionFinalizationCallback
from mlops.hset_state import (
    Ft4Args,
    capture_git,
    create_session_dir,
    extract_runconfig,
    inject_paths,
    persist_config,
    read_session_runconfig,
    read_sessions,
    write_session_runconfig,
)
from mlops.prepare import prepare_run
from mlops.wandb_tracker import make_wandb_logger


_TRAIN_METRIC_BASES = ("train_ce", "train_loss", "loss")
_VAL_METRIC_BASES = ("val_ce", "val_loss")
# Suffix priority: Lightning emits `<name>_epoch` and `<name>_step` when both
# on_step and on_epoch are True; the epoch-aggregated form is what students
# expect when they see "train_ce" in the briefing.
_METRIC_SUFFIX_PRIORITY = ("_epoch", "", "_step")


def _would_be_noop(sessions: list[dict], trainer: Any) -> bool:
    """True if the most recent completed session has already reached the
    trainer's stopping criteria, or if training is disabled (max_epochs <= 0).

    Reads sessions.jsonl (via the `sessions` list) rather than peeking
    into checkpoint files — keeps the audit trail authoritative and avoids
    coupling to Lightning's checkpoint format.

    A previous errored or interrupted session does NOT count: re-running
    might make progress where the previous attempt didn't.
    """
    max_epochs = getattr(trainer, "max_epochs", None)
    max_steps = getattr(trainer, "max_steps", None)

    # If training is disabled (max_epochs == 0), it's a no-op. Lightning
    # uses -1 to mean "unlimited" for both max_epochs and max_steps, so
    # don't treat negative as "disabled".
    if max_epochs == 0:
        return True
    if max_steps == 0:
        return True

    completed = [s for s in sessions if s.get("status") == "completed"]
    if not completed:
        return False
    last = completed[-1]

    last_epochs = last.get("epochs", 0)
    last_steps = last.get("steps", 0)

    # Lightning's max_epochs/max_steps default of -1 means "unlimited" —
    # never treat that as a reachable target.
    if max_epochs is not None and max_epochs > 0 and last_epochs >= max_epochs:
        return True
    if max_steps is not None and max_steps > 0 and last_steps >= max_steps:
        return True
    return False


def run(ft4_args: Ft4Args, *, runs_root: Path = Path("runs")) -> int:
    """Execute the smart flow. Returns 0 on success or graceful abort.

    Exceptions from trainer.fit() (KeyboardInterrupt, training errors)
    propagate to the caller after Lightning's teardown — which runs the
    SessionFinalizationCallback that records the session as 'interrupted'
    or 'error'.
    """
    # ── Phase 0: non-destructive setup ──
    p = prepare_run(ft4_args, runs_root=runs_root)
    cli = p.cli
    version, hset = p.version, p.hset
    ckpt_path = p.ckpt_path
    sessions = p.sessions

    curr_runconfig = extract_runconfig(p.config_dict)
    prev_runconfig, prev_session_id = _read_prev_runconfig(hset.dir)

    briefing_kwargs = dict(
        model_file=ft4_args.model_file,
        version=version,
        hset=hset,
        sessions=sessions,
        ckpt_path=ckpt_path,
        pl_module=cli.model,
        datamodule=cli.datamodule,
        args=ft4_args,
        trainer=p.trainer_summary,
        config=p.config_dict,
        runs_root=runs_root,
        mode=p.mode,
        latest_version=p.latest_version,
        latest_hset=p.latest_hset,
        latest_step=p.latest_step,
        curr_runconfig=curr_runconfig,
        prev_runconfig=prev_runconfig,
        prev_session_id=prev_session_id,
    )
    briefing_panel = render_briefing(**briefing_kwargs)
    briefing_plain = render_briefing_plain(**briefing_kwargs)
    console = Console(stderr=True)

    # No-op short-circuit. If the most recent completed session already
    # reached the trainer's stopping criteria, there's nothing to do. Exit
    # gracefully before Phase 2 — don't create an empty session dir.
    if _would_be_noop(sessions, cli.trainer):
        print(
            "Nothing to do — already at target. "
            "Bump --trainer.max_epochs or --trainer.max_steps to continue.",
            file=sys.stderr,
        )
        return 0

    # ── Phase 2: destructive work ──
    persist_config(cli.parser, cli.config, hset.dir)

    # Inject prompts onto any callback that wants them (e.g., sample callback).
    for cb in cli.trainer.callbacks:
        if hasattr(cb, "prompts"):
            cb.prompts = ft4_args.prompt

    # Create the session dir and print the session/wandb indicators FIRST,
    # before the (potentially slow) baseline sample generation — so the
    # student has the run URL while baseline generation is running.
    session_dir = create_session_dir(hset.dir)
    console.print(f"  [bold]session:[/bold] {session_dir.name}")

    # Optional wandb logger. Opt-in via --wandb; make_wandb_logger raises
    # Ft4UserError with a helpful message if wandb is missing, the user
    # isn't logged in, or initialization fails for any other reason.
    # Attach before inject_paths so the logger sees its session_dir like
    # CSVLogger does.
    if ft4_args.wandb:
        wandb_logger = make_wandb_logger(
            ft4_args.model_file.stem, version, hset, session_dir,
        )
        cli.trainer.loggers.append(wandb_logger)
        # Bold "wandb:" + underlined URL — modern terminals render the
        # underlined URL as Ctrl/Cmd+Click-able.
        url = wandb_logger.experiment.url
        console.print(f"  [bold]wandb:[/bold] [underline]{url}[/underline]")

    inject_paths(cli.trainer, hset.dir, session_dir)

    # Force CSVLogger (and any other lazy-init logger) to inspect its
    # save_dir while it's still empty. Otherwise the dir-not-empty warning
    # fires on the first metric log once runconfig.yaml is sitting there.
    for logger in cli.trainer.loggers:
        try:
            _ = logger.experiment
        except Exception:
            # Best-effort: a logger that can't init shouldn't block the
            # session. The next log call will surface the real error.
            pass

    try:
        write_session_runconfig(session_dir, p.config_dict)
    except Exception as e:
        print(
            f"Warning: runconfig.yaml write failed "
            f"({type(e).__name__}: {e}); session will proceed without it.",
            file=sys.stderr,
        )

    # Phase 1 baseline. UX nicety, not correctness-critical: if generation
    # or write_sample_file raises, training proceeds and the session is
    # still recorded. Non-generative models silently skip (with a warning).
    if ckpt_path is None:
        from mlops.sampling import generate_sample_markdown, write_sample_file
        try:
            baseline_text = generate_sample_markdown(
                cli.model,
                prompts=ft4_args.prompt,
                step=0,
                max_to_generate=60,
            )
            write_sample_file(hset.dir, step=0, text=baseline_text)
        except Exception as e:
            print(
                f"Warning: phase-1 baseline sample generation failed "
                f"({type(e).__name__}: {e}); continuing without baseline samples.",
                file=sys.stderr,
            )

    # Wire per-run state into our callbacks.
    git_commit, git_dirty = capture_git(ft4_args.model_file.parent)
    started_at = datetime.now(timezone.utc)
    for cb in cli.trainer.callbacks:
        if isinstance(cb, SessionFinalizationCallback):
            cb.start(started_at, git_commit, git_dirty, ft4_args.rdesc)
        elif isinstance(cb, BriefingPrintCallback):
            cb.set_briefing(briefing_panel, briefing_plain)

    # torch.compile wraps the model into an optimized graph. Performance-only
    # toggle (no weight changes), so it's not part of the config hash. Applied
    # here, not in prepare_run, because compile is only meaningful for `train`
    # — not for `generate` or `show`, where the one-shot startup cost dwarfs
    # any speedup.
    if ft4_args.compile:
        import torch
        cli.model = torch.compile(cli.model)

    # Fit. On clean exit, on_fit_end fires; on Ctrl-C or error, on_exception
    # fires. Both paths land in SessionFinalizationCallback._finalize before
    # the exception (if any) propagates to our caller.
    #
    # The try/finally ensures we always print a one-line summary, even on
    # Ctrl-C or a training error: Lightning's teardown finalizes the session
    # before the exception unwinds through this frame.
    try:
        cli.trainer.fit(
            cli.model,
            cli.datamodule,
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )
    finally:
        _print_session_summary(hset.dir, session_dir, console)

    return 0


def _read_prev_runconfig(hset_dir: Path) -> tuple[dict | None, str | None]:
    """Find the most-recent prior session's runconfig.yaml.

    Returns `(runconfig_dict, session_name)` or `(None, None)` when there's
    no prior session or the file is missing. Used to drive the briefing's
    runconfig-diff continuation line ('vs s001: lr 1e-4->3e-4').
    """
    sessions_dir = hset_dir / "sessions"
    if not sessions_dir.is_dir():
        return None, None
    subdirs = sorted(
        (p for p in sessions_dir.iterdir() if p.is_dir() and p.name.startswith("s")),
        key=lambda p: p.name,
    )
    if not subdirs:
        return None, None
    latest = subdirs[-1]
    rc = read_session_runconfig(latest)
    return rc, latest.name


def _print_session_summary(hset_dir: Path, session_dir: Path, console: Console) -> None:
    """Print one line describing how this session ended.

    Reads the just-finalized session record from sessions.jsonl and prints
    `session sNNN <status>  epochs=N steps=K  train=X.XXX val=X.XXX  ~Ts/epoch`.
    Fields missing from the record are silently omitted.

    If finalization didn't write the expected session, return without
    printing — better silent than misleading.
    """
    try:
        sessions = read_sessions(hset_dir)
    except Exception:
        return
    if not sessions:
        return
    last = sessions[-1]
    if last.get("session_name") != session_dir.name:
        # Finalization didn't write a record for this session (rare: setup
        # error before fit attempted, or finalization itself crashed).
        return

    parts: list[str] = []
    sid = str(last.get("session_name", ""))
    status_raw = str(last.get("status", ""))
    status = {"completed": "done", "interrupted": "interrupted", "error": "errored"}.get(
        status_raw, status_raw or "done"
    )
    parts.append(f"[bold]session {sid}[/bold] {status}")

    progress: list[str] = []
    epochs = last.get("epochs")
    if isinstance(epochs, int):
        progress.append(f"epochs={epochs}")
    steps = last.get("steps")
    if isinstance(steps, int):
        progress.append(f"steps={steps}")
    if progress:
        parts.append(" ".join(progress))

    metrics: list[str] = []
    for prefix in ("train", "val"):
        key = last.get(f"{prefix}_metric_key")
        val = last.get(f"{prefix}_metric_value")
        if isinstance(key, str) and isinstance(val, (int, float)):
            # Display key strips Lightning's `_epoch` suffix for readability.
            label = key[: -len("_epoch")] if key.endswith("_epoch") else key
            metrics.append(f"{label}={val:.3f}")
    if metrics:
        parts.append(" ".join(metrics))

    est = last.get("est_sec_per_epoch")
    if isinstance(est, (int, float)) and est > 0:
        parts.append(f"~{_humanize_seconds(float(est))}/epoch")

    console.print("  " + "  ".join(parts))


# ── Session record assembly (used by SessionFinalizationCallback) ─────────

def _build_session_record_enrichment(trainer: Any) -> dict:
    """The session-record fields that need a live trainer.

    Read by SessionFinalizationCallback._finalize as a best-effort
    enrichment pass over a minimal record assembled from callback state.
    May fail (e.g., trainer is in a half-state from an in-fit exception);
    the caller wraps in try/except and falls back to the minimal record.

    Records the chosen metric *key* alongside its value: the briefing reads
    `train_metric_key`/`val_metric_key` to label the line correctly (so a
    student sees `val_ce=1.041`, not the misleading generic `val=1.041`).
    """
    metrics = _flatten_callback_metrics(trainer.callback_metrics)
    train_pick = _pick_metric(metrics, _TRAIN_METRIC_BASES)
    val_pick = _pick_metric(metrics, _VAL_METRIC_BASES)
    record: dict = {
        "epochs": trainer.current_epoch,
        "steps": trainer.global_step,
        "all_metrics": metrics,
    }
    if train_pick is not None:
        record["train_metric_key"], record["train_metric_value"] = train_pick
    if val_pick is not None:
        record["val_metric_key"], record["val_metric_value"] = val_pick
    return record


def _build_session_record(
    *,
    session_dir: Path,
    started_at: datetime,
    ended_at: datetime,
    trainer: Any,
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict:
    """Legacy unified accessor — preserved for the unit tests that exercise
    it directly. The production path (SessionFinalizationCallback) now
    calls the minimal+enrichment split instead.
    """
    return {
        "session_name": session_dir.name,
        "session_index": int(session_dir.name[1:]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        **_build_session_record_enrichment(trainer),
    }


def _flatten_callback_metrics(callback_metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in callback_metrics.items():
        f = _to_float(v)
        if f is not None:
            out[k] = f
    return out


def _to_float(v: Any) -> float | None:
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_metric(
    metrics: dict, base_names: tuple[str, ...]
) -> tuple[str, float] | None:
    """Pick (key, value) for the most canonical match among `base_names`.

    Suffix priority is `_epoch > "" > _step`: Lightning emits suffixed keys
    when a `self.log(...)` call has both `on_step=True` and `on_epoch=True`,
    and the epoch-aggregated value is the one we want to show. Falling back
    to `_step` means we still display something honest (and the suffix in
    the key tells the student it's a single-batch value).
    """
    for base in base_names:
        for suffix in _METRIC_SUFFIX_PRIORITY:
            key = f"{base}{suffix}"
            if key in metrics:
                v = metrics[key]
                if isinstance(v, (int, float)):
                    return key, float(v)
    return None
