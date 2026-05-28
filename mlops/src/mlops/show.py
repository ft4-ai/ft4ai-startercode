"""ft4 show — print the briefing for one hset without training.

Pure filesystem reads (no Lightning, no dataset, no model instantiation),
so this command starts instantly and works on any version — even ones whose
model file has changed since they were created. Same shape as `ft4 list`.

Trade-off vs. building the LightningCLI: the `model` row shows the file
stem instead of `Class  N params`, and the `data` row is omitted. Both
fall out of the renderer's existing `pl_module=None` / `datamodule=None`
handling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from rich.console import Console

from mlops.briefing_display import (
    TrainerSummary,
    render_briefing,
    render_briefing_plain,
)
from mlops.hset_state import (
    Ft4Args,
    Ft4UserError,
    all_hsets,
    all_versions,
    detect_mode,
    extract_runconfig,
    latest_ckpt_step,
    read_sessions,
    resolve_checkpoint,
)
from mlops.run import _read_prev_runconfig


def run_show(args: Ft4Args, *, runs_root: Path = Path("runs")) -> int:
    model_dir = runs_root / args.model_file.stem
    if not model_dir.is_dir():
        raise Ft4UserError(
            f"no runs found for {args.model_file.stem} in {runs_root}"
        )

    versions = all_versions(model_dir, newest_first=True)
    if args.version:
        version = next((v for v in versions if v.name == args.version), None)
        if version is None:
            raise Ft4UserError(
                f"model version {args.version} does not exist in {model_dir}"
            )
    elif versions:
        version = versions[0]
    else:
        raise Ft4UserError(f"no model versions under {model_dir}")

    hsets = all_hsets(version.dir, newest_first=True)
    if args.hset:
        hset = next((h for h in hsets if h.name == args.hset), None)
        if hset is None:
            raise Ft4UserError(
                f"hset {args.hset} does not exist in model version {version.name}"
            )
    elif hsets:
        hset = hsets[0]
    else:
        raise Ft4UserError(f"no hsets under {version.dir}")

    config_path = hset.dir / "config.yaml"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text())
        config_dict = loaded if isinstance(loaded, dict) else {}
    else:
        config_dict = {}
    trainer_cfg = config_dict.get("trainer") or {}
    if not isinstance(trainer_cfg, dict):
        trainer_cfg = {}
    trainer_summary = TrainerSummary(
        max_epochs=trainer_cfg.get("max_epochs"),
        max_steps=trainer_cfg.get("max_steps"),
        val_check_interval=trainer_cfg.get("val_check_interval"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
    )

    ckpt_path = resolve_checkpoint(hset.dir, args.ckpt)
    sessions = read_sessions(hset.dir)

    latest_version = versions[0] if versions else None
    sibling_hsets = (
        all_hsets(latest_version.dir, newest_first=True) if latest_version else []
    )
    latest_hset = sibling_hsets[0] if sibling_hsets else None
    latest_step = latest_ckpt_step(hset.dir)
    mode = detect_mode(
        version=version,
        hset=hset,
        ckpt_path=ckpt_path,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
    )

    curr_runconfig = extract_runconfig(config_dict)
    prev_runconfig, prev_session_id = _read_prev_runconfig(hset.dir)

    kwargs = dict(
        model_file=args.model_file,
        version=version,
        hset=hset,
        sessions=sessions,
        ckpt_path=ckpt_path,
        pl_module=None,
        datamodule=None,
        args=args,
        trainer=trainer_summary,
        config=config_dict,
        runs_root=runs_root,
        mode=mode,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
        curr_runconfig=curr_runconfig,
        prev_runconfig=prev_runconfig,
        prev_session_id=prev_session_id,
    )
    if sys.stdout.isatty():
        Console().print(render_briefing(**kwargs))
    else:
        print(render_briefing_plain(**kwargs))
    return 0
