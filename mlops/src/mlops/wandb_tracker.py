"""Weights & Biases (wandb) experiment tracking for ft4.

Single point of contact between ft4 and wandb. The rest of the CLI calls
`make_wandb_logger` here; no other module imports the `wandb` package
directly.

Wandb is opt-in via `--wandb`. When opted in, any failure raises
`Ft4UserError` with an actionable message (and the suggestion to drop
the flag). The errors handled specially:
  - wandb not installed
  - user not logged in (no WANDB_API_KEY, no ~/.netrc entry)
  - WandbLogger / wandb.init() raising for any other reason

ft4's hierarchy is mapped onto wandb as:

  project = <model stem>            (e.g. "transformer")
  group   = <version>/<hset>        (e.g. "v001/h001")
  run     = <version>/<hset>/<session>   (one run per trainer.fit())
  tags    = [<version>, <hset>]
  config  = ft4 hashes + names (for cross-run correlation)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Suppress wandb's multi-line startup banner. We print our own one-line
# indicator from run.py once the run is created. Must be set BEFORE any
# `import wandb` runs anywhere in the process; this module is imported
# from run.py during ft4 startup, so it's the right place.
os.environ.setdefault("WANDB_SILENT", "true")

from mlops.hset_state import Ft4UserError, HsetInfo, VersionInfo


_DROP_FLAG_HINT = "Or drop the --wandb flag to run without wandb."


def make_wandb_logger(
    model_stem: str,
    version: VersionInfo,
    hset: HsetInfo,
    session_dir: Path,
) -> Any:
    """Return a `WandbLogger`. Raises `Ft4UserError` with a helpful message
    if wandb is unavailable, the user isn't logged in, or `WandbLogger`
    initialization fails for any other reason."""
    try:
        import wandb
    except ImportError:
        raise Ft4UserError(
            f"wandb is not installed. Install it with `uv pip install wandb`. {_DROP_FLAG_HINT}"
        )

    if not wandb.api.api_key:
        raise Ft4UserError(
            f"wandb is not logged in. Run `wandb login` first. {_DROP_FLAG_HINT}"
        )

    from lightning.pytorch.loggers import WandbLogger
    try:
        logger = WandbLogger(
            project=model_stem,
            name=f"{version.name}/{hset.name}/{session_dir.name}",
            group=f"{version.name}/{hset.name}",
            tags=[version.name, hset.name],
            config={
                "ft4_version": version.name,
                "ft4_hset": hset.name,
                "ft4_session": session_dir.name,
                "config_hash": hset.config_hash,
                "state_dict_hash": version.state_dict_hash,
                "model_file_hash": version.model_file_hash,
            },
        )
        # Force wandb.init() now (it normally fires lazily on first metric
        # log). This surfaces auth/network failures HERE — where we can
        # raise cleanly — instead of mid-`trainer.fit()`.
        _ = logger.experiment
    except Exception as e:
        raise Ft4UserError(
            f"wandb failed to initialize: {type(e).__name__}: {e}. "
            f"{_DROP_FLAG_HINT}"
        )
    return logger
