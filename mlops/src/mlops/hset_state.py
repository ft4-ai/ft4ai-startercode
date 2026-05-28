"""Hset state, signals, and versioning for ft4.

Layer-1 module: pure Python + torch + stdlib. No Lightning at module load
(Lightning imports are lazy, inside `inject_paths`). Filesystem layout lives
here; routing decisions live here; JSON I/O lives here.

An "hset" (hyperparameter set) is a group of training sessions that share
the same scientifically-meaningful hyperparameters within a model version.
Picking a new hset means training from scratch with different hparams.

Spec references:
  § 6.1 Version resolution
  § 6.2 Hset resolution
  § 6.3 Checkpoint resolution
  § 7   Three signals
  § 11  Schemas (version.json, hset.json, sessions.jsonl)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch.nn as nn


# ── Constants ──────────────────────────────────────────────────────────────

# Keys that belong to the *runconfig* (training-recipe) rather than the hset
# (scientific identity). One set, dual use:
#   - hash_config excludes these prefixes from the hset hash (so changing LR
#     doesn't fork a new hset).
#   - extract_runconfig captures these prefixes into <session>/runconfig.yaml
#     (so each session records the recipe it was run with).
# A black-list-style misses model-specific names (e.g. an optimizer named
# `optimizer_lr` won't be excluded automatically). When that comes up, extend.
RUNCONFIG_STD_KEYS: frozenset[str] = frozenset({
    # Stopping criteria — when to stop training, not what training is.
    "trainer.max_epochs",
    "trainer.max_steps",
    "trainer.max_time",
    "trainer.val_check_interval",
    "trainer.check_val_every_n_epoch",
    "trainer.log_every_n_steps",
    # Infrastructure — observers, not actors.
    "trainer.callbacks",
    "trainer.logger",
    # UI — doesn't affect training results.
    "trainer.enable_progress_bar",
    "trainer.enable_model_summary",
    # Path-bound — ft4 injects these per-session.
    "trainer.default_root_dir",
    # Debugging / dev modes — not training behavior.
    "trainer.fast_dev_run",
    "trainer.detect_anomaly",
    "trainer.profiler",
    "trainer.barebones",
    # Performance autotuning — not scientific.
    "trainer.benchmark",
    # Numerical precision (bf16/fp16/fp32) — recipe, not identity.
    "trainer.precision",
    # Random seed — different runs of the same experiment, not a fork.
    "seed_everything",
    # jsonargparse stores --config paths under `config`; the path list varies
    # by working directory and isn't scientifically meaningful.
    "config",
    # Data-loader infra — affects throughput, not what the model learns.
    # Subclass mode (subclass_mode_data=True) nests these under .init_args.;
    # the bare-name variants are fallbacks for non-subclass data modules.
    "data.init_args.batch_size",
    "data.init_args.num_workers",
    "data.init_args.pin_memory",
    "data.init_args.persistent_workers",
    "data.batch_size",
    "data.num_workers",
    "data.pin_memory",
    "data.persistent_workers",
    # Optimizer / scheduler hparams — the training recipe, not the experiment.
    # Both bare-name and `.init_args.`-nested forms (LightningCLI's
    # subclass_mode_model defaults to False, so model hparams live at model.X,
    # but we hedge for projects that flip subclass_mode_model on).
    "model.lr",
    "model.learning_rate",
    "model.lr_schedule",
    "model.warmup_steps",
    "model.warmup",
    "model.optimizer",
    "model.beta1",
    "model.beta2",
    "model.eps",
    "model.weight_decay",
    "model.init_args.lr",
    "model.init_args.learning_rate",
    "model.init_args.lr_schedule",
    "model.init_args.warmup_steps",
    "model.init_args.warmup",
    "model.init_args.optimizer",
    "model.init_args.beta1",
    "model.init_args.beta2",
    "model.init_args.eps",
    "model.init_args.weight_decay",
    # Dropout is regularization, not architecture: it doesn't change tensor
    # shapes (ckpt remains loadable) and is commonly toggled per session
    # (e.g. drop for fine-tuning). Same principle as LR.
    "model.dropout",
    "model.init_args.dropout",
})

WRITE_SUBCOMMANDS: frozenset[str] = frozenset({"train"})

_VERSION_RE = re.compile(r"^v(\d{3,})$")
_HSET_RE = re.compile(r"^h(\d{3,})$")
_SESSION_RE = re.compile(r"^s(\d{3,})$")
# ft4 configures ModelCheckpoint with `filename='step_{step:06d}'` →
# 'step_000123.ckpt'. Legacy run directories use Lightning's default
# `filename='{step}'` → 'step=123.ckpt'. The character class accepts both.
_CKPT_STEP_RE = re.compile(r"^step[=_](\d+)\.ckpt$")


# ── Mode ───────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    """Briefing mode. str-valued so it serializes cleanly into log records
    and is comparable to literals in tests."""
    FRESH = "fresh"
    RESUME = "resume"
    BACKTRACK = "backtrack"


# ── Exceptions ─────────────────────────────────────────────────────────────

class Ft4UserError(Exception):
    """An error message intended for the user; main.py prints to stderr
    and exits with status 2."""


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ft4Args:
    """Parsed CLI args. Held as a single dataclass to avoid passing many
    individual parameters through routing functions."""
    subcommand: str
    model_file: Path
    model_class: str | None = None
    hset: str | None = None
    version: str | None = None
    ckpt: str = "last"
    new_version: bool = False
    new_hset: bool = False
    vdesc: str | None = None
    hdesc: str | None = None
    rdesc: str | None = None
    prompt: list[str] | None = None
    compile: bool = False
    wandb: bool = False
    # Generate-only. `None` means "use LangGen's default" (temperature/min_p)
    # or the CLI fallback (max_tokens).
    temperature: float | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    force: bool = False
    # List-only.
    sort_by: str = "time"           # "time" or "perf"
    limit: int | None = None
    lightning_args: list[str] = field(default_factory=list)

    @property
    def is_write_op(self) -> bool:
        return self.subcommand in WRITE_SUBCOMMANDS


@dataclass(frozen=True)
class Signals:
    """Three hashes computed every invocation. § 7."""
    state_dict_hash: str
    model_file_hash: str
    config_hash: str


@dataclass(frozen=True)
class VersionInfo:
    name: str           # "vNNN"
    dir: Path
    state_dict_hash: str
    model_file_hash: str
    description: str | None = None
    created_at: str | None = None   # ISO timestamp from version.json


@dataclass(frozen=True)
class HsetInfo:
    name: str           # "hNNN"
    dir: Path
    config_hash: str
    description: str | None = None


# ── Hashing ────────────────────────────────────────────────────────────────

def hash_state_dict(model: nn.Module) -> str:
    """SHA-256 of sorted (name, shape, dtype) tuples from state_dict.
    Includes both parameters and buffers."""
    items = sorted(
        (name, tuple(t.shape), str(t.dtype))
        for name, t in model.state_dict().items()
    )
    payload = json.dumps(items).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted-key form. Lists are kept as-is
    (single key with list value); exclusion uses prefix matching so a
    single entry like 'trainer.callbacks' filters the whole list."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def hash_config(config_dict: dict, exclude_prefixes: frozenset[str]) -> str:
    """Hash of the resolved config dict with `exclude_prefixes`-matching
    keys removed. Matching is by exact equality OR dotted-prefix: an entry
    'trainer.logger' filters 'trainer.logger', 'trainer.logger.class_path',
    'trainer.logger.init_args.save_dir', etc.
    """
    flat = _flatten(config_dict)
    filtered = {
        k: v for k, v in flat.items()
        if not any(k == p or k.startswith(p + ".") for p in exclude_prefixes)
    }
    payload = json.dumps(filtered, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_signals(model: nn.Module, model_file: Path, config_dict: dict) -> Signals:
    return Signals(
        state_dict_hash=hash_state_dict(model),
        model_file_hash=hash_file(model_file),
        config_hash=hash_config(config_dict, RUNCONFIG_STD_KEYS),
    )


# ── Version resolution ─────────────────────────────────────────────────────

def resolve_version(signals: Signals, args: Ft4Args, runs_root: Path) -> VersionInfo:
    """§ 6.1. Returns a model version (creating if needed for write ops;
    erroring if no compatible one exists for read ops)."""
    model_dir = runs_root / args.model_file.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.is_write_op:
        return _resolve_version_write(model_dir, signals, args)
    return _resolve_version_read(model_dir, signals, args)


def _resolve_version_write(model_dir: Path, signals: Signals, args: Ft4Args) -> VersionInfo:
    if args.new_version:
        return _create_new_version(model_dir, signals, args)

    if args.version:
        v = _load_named_version(model_dir, args.version)
        if v.model_file_hash != signals.model_file_hash:
            raise Ft4UserError(
                f"--version {args.version}: model file has changed; "
                f"use --new-version to fork"
            )
        return _maybe_update_version_desc(v, args.vdesc)

    # A version is the model's *code* identity. Different parameter shapes
    # under the same source file are different hsets, not versions. So
    # we match by model_file_hash only and iterate every existing version
    # newest-first — the user may have an older version that matches even
    # if the most recent one was forked manually.
    for v in all_versions(model_dir, newest_first=True):
        if v.model_file_hash == signals.model_file_hash:
            return _maybe_update_version_desc(v, args.vdesc)
    return _create_new_version(model_dir, signals, args)


def _resolve_version_read(model_dir: Path, signals: Signals, args: Ft4Args) -> VersionInfo:
    if args.version:
        v = _load_named_version(model_dir, args.version)
        if v.model_file_hash != signals.model_file_hash:
            raise Ft4UserError(
                f"--version {args.version}: model file has changed "
                f"since this version was created"
            )
        return v
    for v in all_versions(model_dir, newest_first=True):
        if v.model_file_hash == signals.model_file_hash:
            return v
    raise Ft4UserError(
        "no compatible model version found for the current model file; "
        "train first or use --version to pin an older version"
    )


def _latest_version(model_dir: Path) -> VersionInfo | None:
    versions = all_versions(model_dir, newest_first=True)
    return versions[0] if versions else None


def all_versions(model_dir: Path, newest_first: bool = True) -> list[VersionInfo]:
    if not model_dir.exists():
        return []
    dirs = [
        p for p in model_dir.iterdir()
        if p.is_dir() and _VERSION_RE.match(p.name) and (p / "version.json").exists()
    ]
    # Tiebreaker on name keeps the order deterministic when mtimes match
    # (common in tests that build fixtures in a tight loop).
    dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=newest_first)
    return [_load_version(p) for p in dirs]


def _load_version(version_dir: Path) -> VersionInfo:
    data = json.loads((version_dir / "version.json").read_text())
    return VersionInfo(
        name=version_dir.name,
        dir=version_dir,
        state_dict_hash=data["state_dict_hash"],
        model_file_hash=data["model_file_hash"],
        description=data.get("description"),
        created_at=data.get("created_at"),
    )


def _load_named_version(model_dir: Path, name: str) -> VersionInfo:
    version_dir = model_dir / name
    if not version_dir.exists() or not (version_dir / "version.json").exists():
        raise Ft4UserError(f"model version {name} does not exist in {model_dir}")
    return _load_version(version_dir)


def _create_new_version(model_dir: Path, signals: Signals, args: Ft4Args) -> VersionInfo:
    existing = [
        p.name for p in model_dir.iterdir()
        if p.is_dir() and _VERSION_RE.match(p.name)
    ]
    next_n = max((int(_VERSION_RE.match(n).group(1)) for n in existing), default=0) + 1
    name = f"v{next_n:03d}"
    version_dir = model_dir / name
    version_dir.mkdir()

    git_commit, git_dirty = capture_git(args.model_file.parent)
    created_at = _now_iso()
    data = {
        "created_at": created_at,
        "description": args.vdesc,
        "state_dict_hash": signals.state_dict_hash,
        "model_file_hash": signals.model_file_hash,
        "git_commit_at_creation": git_commit,
        "git_dirty_at_creation": git_dirty,
    }
    (version_dir / "version.json").write_text(json.dumps(data, indent=2))
    (version_dir / "model.py.snapshot").write_bytes(args.model_file.read_bytes())

    return VersionInfo(
        name=name,
        dir=version_dir,
        state_dict_hash=signals.state_dict_hash,
        model_file_hash=signals.model_file_hash,
        description=args.vdesc,
        created_at=created_at,
    )


def _maybe_update_version_desc(version: VersionInfo, vdesc: str | None) -> VersionInfo:
    if vdesc is None or vdesc == version.description:
        return version
    data = json.loads((version.dir / "version.json").read_text())
    data["description"] = vdesc
    (version.dir / "version.json").write_text(json.dumps(data, indent=2))
    return VersionInfo(
        name=version.name,
        dir=version.dir,
        state_dict_hash=version.state_dict_hash,
        model_file_hash=version.model_file_hash,
        description=vdesc,
        created_at=version.created_at,
    )


# ── Hset resolution ────────────────────────────────────────────────────────

def resolve_hset(version: VersionInfo, signals: Signals, args: Ft4Args) -> HsetInfo:
    """§ 6.2. Returns the matching hset within `version`, creating one if
    no config_hash match exists (write ops) or erroring (--hset pin on
    read ops with no match).

    `--new-hset` forces a fresh hset even when the current config hashes to
    match an existing one — useful for branching off an old checkpoint
    without overwriting the original line of training.
    """
    if args.hset:
        return _load_named_hset(version, args.hset, signals, args)
    if args.new_hset and args.is_write_op:
        return _create_new_hset(version, signals, args)
    for h in all_hsets(version.dir, newest_first=True):
        if h.config_hash == signals.config_hash:
            return _maybe_update_hset_desc(h, args.hdesc)
    return _create_new_hset(version, signals, args)


def all_hsets(version_dir: Path, newest_first: bool = True) -> list[HsetInfo]:
    if not version_dir.exists():
        return []
    dirs = [
        p for p in version_dir.iterdir()
        if p.is_dir() and _HSET_RE.match(p.name) and (p / "hset.json").exists()
    ]
    # Tiebreaker on name keeps the order deterministic when mtimes match.
    dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=newest_first)
    return [_load_hset(p) for p in dirs]


def _load_hset(hset_dir: Path) -> HsetInfo:
    data = json.loads((hset_dir / "hset.json").read_text())
    return HsetInfo(
        name=hset_dir.name,
        dir=hset_dir,
        config_hash=data["config_hash"],
        description=data.get("description"),
    )


def _load_named_hset(
    version: VersionInfo, name: str, signals: Signals, args: Ft4Args
) -> HsetInfo:
    hset_dir = version.dir / name
    if not hset_dir.exists() or not (hset_dir / "hset.json").exists():
        raise Ft4UserError(
            f"hset {name} does not exist in model version {version.name}"
        )
    hset = _load_hset(hset_dir)
    if args.is_write_op and hset.config_hash != signals.config_hash:
        # ft4 normally auto-loads <hset>/config.yaml when --hset is given,
        # so reaching here means the saved config no longer reproduces the
        # recorded hash (corrupt file, ft4 version skew, or runs/ rebuilt
        # by hand). Tell the student the two ways out.
        raise Ft4UserError(
            f"--hset {name}: loaded config does not reproduce {name}'s "
            f"recorded hash ({hset.config_hash[:8]} vs current "
            f"{signals.config_hash[:8]}). Either delete "
            f"{hset_dir} and retrain, or pass --new-hset to fork from "
            f"your current config."
        )
    return _maybe_update_hset_desc(hset, args.hdesc)


def _create_new_hset(version: VersionInfo, signals: Signals, args: Ft4Args) -> HsetInfo:
    existing = [
        p.name for p in version.dir.iterdir()
        if p.is_dir() and _HSET_RE.match(p.name)
    ]
    next_n = max((int(_HSET_RE.match(n).group(1)) for n in existing), default=0) + 1
    name = f"h{next_n:03d}"
    hset_dir = version.dir / name
    hset_dir.mkdir()
    (hset_dir / "checkpoints").mkdir()

    now = _now_iso()
    data = {
        "created_at": now,
        "last_session_at": now,
        "description": args.hdesc,
        "config_hash": signals.config_hash,
    }
    (hset_dir / "hset.json").write_text(json.dumps(data, indent=2))

    return HsetInfo(
        name=name,
        dir=hset_dir,
        config_hash=signals.config_hash,
        description=args.hdesc,
    )


def _maybe_update_hset_desc(hset: HsetInfo, hdesc: str | None) -> HsetInfo:
    if hdesc is None or hdesc == hset.description:
        return hset
    data = json.loads((hset.dir / "hset.json").read_text())
    data["description"] = hdesc
    (hset.dir / "hset.json").write_text(json.dumps(data, indent=2))
    return HsetInfo(
        name=hset.name,
        dir=hset.dir,
        config_hash=hset.config_hash,
        description=hdesc,
    )


def update_hset_last_session(hset_dir: Path, when: datetime) -> None:
    data = json.loads((hset_dir / "hset.json").read_text())
    data["last_session_at"] = when.isoformat()
    (hset_dir / "hset.json").write_text(json.dumps(data, indent=2))


# ── Checkpoint resolution ──────────────────────────────────────────────────

def resolve_checkpoint(hset_dir: Path, choice: str) -> Path | None:
    """§ 6.3. Return Path to checkpoint to load, or None if there's none yet.

    Caller decides what None means: for run/train, train from scratch; for
    read ops (generate/validate), error out.

    V1: 'last' or explicit path. 'best' is deferred to V2.
    """
    if choice == "last":
        last = hset_dir / "checkpoints" / "last.ckpt"
        return last if last.exists() else None
    if choice == "best":
        raise Ft4UserError(
            "'--ckpt best' is deferred to V2; use 'last' or an explicit path"
        )
    path = Path(choice)
    if not path.exists():
        raise Ft4UserError(f"checkpoint not found: {path}")
    return path


def parse_ckpt_step(ckpt_path: Path | None) -> int | None:
    """Pull the integer step from a `step=N.ckpt` filename, or from the
    symlink target (so `last.ckpt -> step=N.ckpt` resolves to N).

    Returns None for anything that doesn't match the Lightning convention.
    """
    if ckpt_path is None:
        return None
    # If it's a symlink, follow it — `last.ckpt` is the common case.
    try:
        if ckpt_path.is_symlink():
            target = Path(ckpt_path.readlink())
            m = _CKPT_STEP_RE.match(target.name)
            if m:
                return int(m.group(1))
    except (OSError, ValueError):
        pass
    m = _CKPT_STEP_RE.match(ckpt_path.name)
    return int(m.group(1)) if m else None


def latest_ckpt_step(hset_dir: Path) -> int | None:
    """Return the highest step number among `<hset>/checkpoints/step=N.ckpt`
    files, or None if there are none. Used by detect_mode to decide whether
    a loaded checkpoint is the latest (=> resume) or older (=> backtrack)."""
    ckpt_dir = hset_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    steps: list[int] = []
    for p in ckpt_dir.iterdir():
        m = _CKPT_STEP_RE.match(p.name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def detect_mode(
    *,
    version: VersionInfo,
    hset: HsetInfo,
    ckpt_path: Path | None,
    latest_version: VersionInfo | None,
    latest_hset: HsetInfo | None,
    latest_step: int | None,
) -> Mode:
    """Decide the briefing mode based on what's *actually loaded* vs latest.

    FRESH: no ckpt loaded.
    BACKTRACK: loaded ckpt OR resolved hset/version is not the latest along
      its respective axis. A user typing `--ckpt step=N.ckpt` where N happens
      to BE the latest is still RESUME, not BACKTRACK — we compare resolved
      identity, not flag syntax.
    RESUME: ckpt loaded, and (resolved == latest) along every axis.
    """
    if ckpt_path is None:
        return Mode.FRESH
    if latest_version is not None and version.name != latest_version.name:
        return Mode.BACKTRACK
    if latest_hset is not None and hset.name != latest_hset.name:
        return Mode.BACKTRACK
    loaded_step = parse_ckpt_step(ckpt_path)
    if (loaded_step is not None and latest_step is not None
            and loaded_step != latest_step):
        return Mode.BACKTRACK
    return Mode.RESUME


# ── Session log ────────────────────────────────────────────────────────────

def append_session(hset_dir: Path, record: dict) -> None:
    with (hset_dir / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_sessions(hset_dir: Path) -> list[dict]:
    path = hset_dir / "sessions.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def diff_vs_prior_hset(
    version: VersionInfo, hset: HsetInfo, this_config: dict,
) -> tuple[str, list[tuple[str, Any, Any]]] | None:
    """Compare `this_config` (the resolved config dict for `hset` this run)
    against the most recent sibling hset's persisted config.yaml.

    Returns `(sibling_name, diffs)` where each diff is
    `(dotted_key, prior_val, this_val)`. Returns None when no sibling
    exists, or the sibling's config.yaml is missing or unparseable.
    `RUNCONFIG_STD_KEYS` are filtered out — only scientifically
    meaningful keys appear in the diff.
    """
    import yaml

    prior_hset = None
    for sibling in all_hsets(version.dir, newest_first=True):
        if sibling.name != hset.name:
            prior_hset = sibling
            break
    if prior_hset is None:
        return None

    prior_path = prior_hset.dir / "config.yaml"
    if not prior_path.exists():
        return None
    try:
        prior_config = yaml.safe_load(prior_path.read_text()) or {}
    except yaml.YAMLError:
        return None

    this_flat = _flatten(this_config)
    prior_flat = _flatten(prior_config)

    diffs: list[tuple[str, Any, Any]] = []
    for key in sorted(set(this_flat.keys()) | set(prior_flat.keys())):
        if any(key == p or key.startswith(p + ".") for p in RUNCONFIG_STD_KEYS):
            continue
        prior_v = prior_flat.get(key)
        this_v = this_flat.get(key)
        if prior_v != this_v:
            diffs.append((key, prior_v, this_v))
    return prior_hset.name, diffs


def diff_hset_configs(
    hset_dir: Path,
    baseline_hset_dir: Path,
    *,
    max_keys: int = 2,
    exclude_prefixes: frozenset[str] = RUNCONFIG_STD_KEYS,
) -> str:
    """Render the config diff for the `ft4 list` Notes column.

    Reads `<hset>/config.yaml` and `<baseline_hset>/config.yaml`, finds keys
    whose values differ (after filtering `exclude_prefixes`), and returns a
    compact `key=value, key=value, +N more` string (new values only).

    Returns `""` when either config.yaml is missing/unparseable, when the
    hsets are the same dir, or when no scientifically meaningful keys differ.
    """
    import yaml

    if hset_dir == baseline_hset_dir:
        return ""
    this_path = hset_dir / "config.yaml"
    base_path = baseline_hset_dir / "config.yaml"
    if not this_path.exists() or not base_path.exists():
        return ""
    try:
        this_config = yaml.safe_load(this_path.read_text()) or {}
        base_config = yaml.safe_load(base_path.read_text()) or {}
    except yaml.YAMLError:
        return ""

    this_flat = _flatten(this_config)
    base_flat = _flatten(base_config)

    diffs: list[tuple[str, Any]] = []
    for key in sorted(set(this_flat.keys()) | set(base_flat.keys())):
        if any(key == p or key.startswith(p + ".") for p in exclude_prefixes):
            continue
        if this_flat.get(key) != base_flat.get(key):
            diffs.append((key, this_flat.get(key)))

    if not diffs:
        return ""
    shown = diffs[:max_keys]
    parts = [f"{_clean_key(k)}={_format_diff_value(v)}" for k, v in shown]
    if len(diffs) > max_keys:
        parts.append(f"+{len(diffs) - max_keys} more")
    return ", ".join(parts)


def _clean_key(key: str) -> str:
    """Strip jsonargparse's ``.init_args.`` boilerplate so display keys
    read as ``data.batch_size`` instead of ``data.init_args.batch_size``.
    """
    return key.replace(".init_args.", ".")


def _format_diff_value(v: Any) -> str:
    """Compact string for a hparam value in the Notes column.

    Floats get `g` formatting so 0.0001 stays readable and 0.33333 doesn't
    blow out width. Everything else uses str(); long values get truncated
    so a stray list/dict can't ruin the row.
    """
    if isinstance(v, float):
        return f"{v:g}"
    s = str(v)
    if len(s) > 20:
        return s[:19] + "…"
    return s


def latest_sibling_session_pace(
    version: VersionInfo, this_hset: HsetInfo,
) -> tuple[str, float] | None:
    """For pace estimation on a fresh hset: find the most recent session
    across sibling hsets that recorded a positive `est_sec_per_epoch`.

    Returns `(sibling_hset_name, est_sec_per_epoch)`, or None if no
    sibling has usable timing data. Sibling hsets share the model file
    (same version) but may differ in shape-affecting hparams, so per-
    epoch time from a sibling is a rough approximation, not exact.
    """
    for sibling in all_hsets(version.dir, newest_first=True):
        if sibling.name == this_hset.name:
            continue
        for sess in reversed(read_sessions(sibling.dir)):
            est = sess.get("est_sec_per_epoch")
            if isinstance(est, (int, float)) and est > 0:
                return sibling.name, float(est)
    return None


# ── Runconfig (per-session training-recipe artifact) ──────────────────────

def extract_runconfig(config_dict: dict) -> dict:
    """Pick out the runconfig-relevant keys from a resolved config dict.

    Returns a sorted dict of flattened dotted keys whose names match
    `RUNCONFIG_STD_KEYS` (exact or dotted-prefix). The values are taken from
    the resolved config as-is — list/dict values pass through.
    """
    flat = _flatten(config_dict)
    out = {
        k: v for k, v in flat.items()
        if any(k == p or k.startswith(p + ".") for p in RUNCONFIG_STD_KEYS)
    }
    return dict(sorted(out.items()))


def write_session_runconfig(session_dir: Path, config_dict: dict) -> None:
    """Persist this session's runconfig.yaml next to its metrics.csv.

    Called at session start (from run.py) so the artifact survives crashes
    and is available for the next run's runconfig-diff line in the briefing.

    Values are coerced through a JSON round-trip with `default=str` so
    jsonargparse-specific types (e.g. its Path wrapper for `--config` paths)
    serialize as plain strings — the same trick `hash_config` uses.
    """
    import yaml
    payload = extract_runconfig(config_dict)
    safe = json.loads(json.dumps(payload, default=str))
    (session_dir / "runconfig.yaml").write_text(
        yaml.safe_dump(safe, sort_keys=True, default_flow_style=False)
    )


def read_session_runconfig(session_dir: Path) -> dict | None:
    """Inverse of write_session_runconfig. Returns None when the file is
    missing or unparseable — callers treat that as 'no diff possible'."""
    import yaml
    path = session_dir / "runconfig.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data


# Runconfig keys whose values are useful in the per-session forensics file
# but pure noise in the briefing diff. `config` is the cascade `--config`
# path list, which changes whenever the user adds/removes a .local.yaml or
# loads an old hset (auto-load substitutes the hset's saved config.yaml
# for <model>.yaml) — neither is a scientifically meaningful change.
_DIFF_HIDDEN_RUNCONFIG_KEYS: frozenset[str] = frozenset({"config"})


def diff_runconfigs(
    curr: dict | None, prev: dict | None,
) -> list[tuple[str, Any, Any]]:
    """Sorted list of `(key, prev_val, curr_val)` for keys that differ.

    Treats either side being None as an empty dict (so a missing prior
    session yields no diffs). Used by the briefing's runconfig row.
    """
    a = curr or {}
    b = prev or {}
    diffs: list[tuple[str, Any, Any]] = []
    for key in sorted(set(a.keys()) | set(b.keys())):
        if key in _DIFF_HIDDEN_RUNCONFIG_KEYS:
            continue
        if a.get(key) != b.get(key):
            diffs.append((key, b.get(key), a.get(key)))
    return diffs


# ── Git capture ────────────────────────────────────────────────────────────

def capture_git(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return (commit_hash, dirty_flag). Both None outside a git repo or
    if git itself is unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode()
        dirty = bool(status.strip())
    except subprocess.CalledProcessError:
        dirty = False
    return commit, dirty


# ── Session directory ──────────────────────────────────────────────────────

def create_session_dir(hset_dir: Path) -> Path:
    """Create and return <hset>/sessions/sNNN/ for the next session.

    NNN is one more than the highest existing s-numbered dir under
    sessions/, zero-padded to 3 digits. Creates the sessions/ parent if
    missing.

    Each training session gets its own subdirectory; Lightning's stock
    CSVLogger writes <session>/metrics.csv there, and
    cumulative_metrics.rebuild_cumulative_metrics merges all sessions
    into <hset>/metrics.csv after training.
    """
    sessions_root = hset_dir / "sessions"
    sessions_root.mkdir(exist_ok=True)
    existing = [
        p.name for p in sessions_root.iterdir()
        if p.is_dir() and _SESSION_RE.match(p.name)
    ]
    next_n = max(
        (int(_SESSION_RE.match(n).group(1)) for n in existing),
        default=0,
    ) + 1
    session_dir = sessions_root / f"s{next_n:03d}"
    session_dir.mkdir()
    return session_dir


# ── Lightning seam (lazy import) ───────────────────────────────────────────

def inject_paths(trainer, hset_dir: Path, session_dir: Path) -> None:
    """Mutate paths on already-instantiated callbacks and logger. Backed by
    contract tests test_modelcheckpoint_dirpath_mutation and
    test_csvlogger_save_dir_mutation.

    Mutations applied:
      - ModelCheckpoint.dirpath  → <hset>/checkpoints   (hset-level)
      - logger._save_dir         → <session_dir>         (per session)
      - any callback with `hset_dir`    → set to <hset_dir>
      - any callback with `session_dir` → set to <session_dir>

    Duck-typing on `hset_dir` / `session_dir` attributes lets us wire
    future callbacks (e.g., SampleGenerationCallback) without coupling
    this module to their class names.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.dirpath = str(hset_dir / "checkpoints")
        if hasattr(cb, "hset_dir"):
            cb.hset_dir = hset_dir
        if hasattr(cb, "session_dir"):
            cb.session_dir = session_dir
    for logger in trainer.loggers:
        logger._save_dir = str(session_dir)


def persist_config(parser, config, hset_dir: Path) -> None:
    """Write resolved config to <hset>/config.yaml. Replaces LightningCLI's
    SaveConfigCallback (which we disable with save_config_callback=None)."""
    (hset_dir / "config.yaml").write_text(parser.dump(config, format="yaml"))


# ── Helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
