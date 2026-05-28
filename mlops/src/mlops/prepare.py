"""Phase 0 setup shared across ft4 subcommands.

`prepare_run(args)` runs the read-mostly part of `ft4 train`'s start-up:

  1. Discover the model class from the .py file.
  2. Build LightningCLI (instantiates model + datamodule + trainer).
  3. Compute the three signals (state_dict_hash, model_file_hash, config_hash).
  4. Resolve version → hset → checkpoint.
  5. Read the hset's session history.

The returned `PreparedRun` is what every read subcommand (`generate`, `show`)
needs to do its job, and what `ft4 train` consumes before Phase 2 (the
destructive training work).

Read-only for non-write subcommands. For write subcommands (`run`), the
underlying `resolve_version` / `resolve_hset` may create a new version or
hset directory — that's intrinsic to write ops, not this module's concern.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlops.briefing_display import TrainerSummary
from mlops.hset_state import (
    Ft4Args,
    HsetInfo,
    Mode,
    Signals,
    VersionInfo,
    all_hsets,
    all_versions,
    compute_signals,
    detect_mode,
    latest_ckpt_step,
    read_sessions,
    resolve_checkpoint,
    resolve_hset,
    resolve_version,
)


@dataclass
class PreparedRun:
    """Everything Phase 0 produces. Not frozen because callers (notably
    `ft4 train`) may attach further per-session state to fields like `trainer`
    after construction."""
    cli: Any                    # lightning.pytorch.cli.LightningCLI
    model: Any                  # cli.model (LightningModule)
    datamodule: Any             # cli.datamodule
    trainer: Any                # cli.trainer (not yet fit)
    config_dict: dict
    signals: Signals
    version: VersionInfo
    hset: HsetInfo
    ckpt_path: Path | None
    sessions: list[dict]
    trainer_summary: TrainerSummary
    # Mode + latest-along-each-axis hints. The renderer uses these to pick
    # title verb, border colour, and the backtrack-row copy. Latest pointers
    # are None when nothing matches (e.g. fresh hset with no siblings).
    mode: Mode
    latest_version: VersionInfo | None
    latest_hset: HsetInfo | None
    latest_step: int | None


def prepare_run(args: Ft4Args, *, runs_root: Path = Path("runs")) -> PreparedRun:
    # Lazy imports to avoid pulling Lightning when callers only want the
    # type. Cheap once Python has cached the import.
    from mlops.main import build_lightning_cli, discover_model_class

    model_class = discover_model_class(args.model_file, args.model_class)
    cli = build_lightning_cli(args, model_class, runs_root=runs_root)
    _ensure_session_finalization(cli.trainer)

    config_dict = cli.config.as_dict()
    signals = compute_signals(cli.model, args.model_file, config_dict)

    version = resolve_version(signals, args, runs_root=runs_root)
    hset = resolve_hset(version, signals, args)
    ckpt_path = resolve_checkpoint(hset.dir, args.ckpt)
    sessions = read_sessions(hset.dir)

    # Mode detection compares resolved identity against the *latest* along
    # each axis. We compute the latest pointers once here so the renderer
    # can also use them for the backtrack-row copy.
    model_dir = runs_root / args.model_file.stem
    versions = all_versions(model_dir, newest_first=True)
    latest_version = versions[0] if versions else None
    sibling_hsets = all_hsets(latest_version.dir, newest_first=True) if latest_version else []
    latest_hset = sibling_hsets[0] if sibling_hsets else None
    latest_step = latest_ckpt_step(hset.dir)
    mode = detect_mode(
        version=version, hset=hset, ckpt_path=ckpt_path,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
    )

    trainer_summary = TrainerSummary(
        max_epochs=getattr(cli.trainer, "max_epochs", None),
        max_steps=getattr(cli.trainer, "max_steps", None),
        val_check_interval=getattr(cli.trainer, "val_check_interval", None),
        gradient_clip_val=getattr(cli.trainer, "gradient_clip_val", None),
    )

    return PreparedRun(
        cli=cli,
        model=cli.model,
        datamodule=cli.datamodule,
        trainer=cli.trainer,
        config_dict=config_dict,
        signals=signals,
        version=version,
        hset=hset,
        ckpt_path=ckpt_path,
        sessions=sessions,
        trainer_summary=trainer_summary,
        mode=mode,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
    )


def _ensure_session_finalization(trainer: Any) -> None:
    """Auto-inject SessionFinalizationCallback if a user's <model>.yaml
    replaced trainer.callbacks without it.

    Without this callback, sessions.jsonl and hset-level cumulative metrics
    never get written — the run looks fine but the audit trail vanishes.
    Quietly re-add it and tell the user, so they know to fix their yaml.
    """
    from mlops.session_finalization import SessionFinalizationCallback

    for cb in trainer.callbacks:
        if isinstance(cb, SessionFinalizationCallback):
            return
    trainer.callbacks.append(SessionFinalizationCallback())
    print(
        "note: SessionFinalizationCallback was missing from trainer.callbacks "
        "(your <model>.yaml likely overrides the list). ft4 has auto-added it "
        "so sessions.jsonl and metrics.csv still get written.",
        file=sys.stderr,
    )
