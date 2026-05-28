"""Tip selector for the ft4 briefing.

The briefing's `tip` row is always present. The selector walks a 4-tier
prioritized list and returns one line:

  * Tier 1 — critical warnings (last session errored, long gap since last run)
  * Tier 2 — setup nudges (missing --hdesc / --vdesc on fresh runs)
  * Tier 3 — teaching nudges (suggest --compile, --wandb, etc.)
  * Tier 4 — pedagogical fallback (always applicable; rotates per session)

Selection within a tier is random (seeded deterministically by
`(version, hset, n_sessions)` so the same state always picks the same tip,
tests stay deterministic, and advancing sessions rotates naturally).
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from mlops.hset_state import Ft4Args, HsetInfo, VersionInfo

# Forward reference; TrainerSummary lives in briefing_display.py to keep its
# definition close to the renderer. We accept it as Any here to avoid an
# import cycle.

_HSET_RE = re.compile(r"^h\d{3,}$")


@dataclass(frozen=True)
class TipContext:
    args: Ft4Args | None
    version: VersionInfo
    hset: HsetInfo
    n_sessions: int
    last_session: dict | None = None
    pl_module: Any = None
    datamodule: Any = None
    trainer_summary: Any = None       # TrainerSummary; Any avoids cycle
    now: datetime | None = None


# ── predicate helpers ─────────────────────────────────────────────────────


def _is_resume(ctx: TipContext) -> bool:
    return ctx.last_session is not None


def _is_fresh_hset(ctx: TipContext) -> bool:
    return ctx.n_sessions == 0


def _days_since(iso: str | None, now: datetime | None) -> float:
    if not isinstance(iso, str):
        return 0.0
    try:
        when = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds() / 86400)


def _only_one_hset(version: VersionInfo, this_hset: HsetInfo) -> bool:
    if not version.dir.exists():
        return True
    siblings = [
        p for p in version.dir.iterdir()
        if p.is_dir() and _HSET_RE.match(p.name) and p.name != this_hset.name
    ]
    return not siblings


# ── tier tables ───────────────────────────────────────────────────────────

TipPredicate = Callable[[TipContext], bool]
TipEntry = tuple[TipPredicate, str]


_TIER1: list[TipEntry] = [
    (
        lambda ctx: _is_resume(ctx) and ctx.last_session.get("status") == "error",
        "last session ended in error — see sessions/sNNN/ for logs",
    ),
    (
        lambda ctx: _is_resume(ctx) and _days_since(ctx.last_session.get("ended_at"), ctx.now) > 7,
        "last session was over a week ago — double-check config.yaml hasn't drifted",
    ),
]

_TIER2: list[TipEntry] = [
    (
        lambda ctx: _is_fresh_hset(ctx) and bool(ctx.args) and ctx.args.hdesc is None,
        'label this hset with --hdesc "..." for future you',
    ),
    (
        lambda ctx: ctx.version.description is None and bool(ctx.args) and ctx.args.vdesc is None,
        'label this version with --vdesc "..." so you can find it later',
    ),
]

_TIER3: list[TipEntry] = [
    (
        lambda ctx: bool(ctx.args) and not ctx.args.compile,
        "use --compile to make training 10–30% faster (once your model works)",
    ),
    (
        lambda ctx: bool(ctx.args) and not ctx.args.wandb,
        "use --wandb to track this run in Weights & Biases",
    ),
    (
        lambda ctx: (
            ctx.pl_module is not None
            and hasattr(ctx.pl_module, "generate_samples")
            and bool(ctx.args)
            and not ctx.args.prompt
        ),
        'use --prompt "..." to seed sample generation with your own text',
    ),
    (
        lambda ctx: (
            ctx.trainer_summary is not None
            and ctx.trainer_summary.val_check_interval == 1.0
        ),
        "use --trainer.val_check_interval=0.25 to validate 4× per epoch",
    ),
    (
        lambda ctx: (
            ctx.trainer_summary is not None
            and ctx.trainer_summary.gradient_clip_val is None
        ),
        "use --trainer.gradient_clip_val=1.0 to stabilise training against loss spikes",
    ),
]

# Tier-4 fallback tips — student MLops nudges, not ft4 internals. The selector
# rotates among them deterministically across sessions (see pick_tip).
_TIER4: list[str] = [
    "use `ft4 list <model.py>` to compare hsets and see your best run",
    "to add more training to this hset, bump --trainer.max_epochs=N and re-run",
    "change a recipe knob (e.g. --model.lr=1e-3) and ft4 keeps the same hset — the diff lands in the runconfig row",
    "change a shape knob (e.g. --model.dim=512) and ft4 forks a new hset automatically",
    "use `ft4 show <model.py> --hset hNNN` to inspect another hset without training",
    "Ctrl-C is safe — finalization runs on exit and records the session as 'interrupted'",
    "metrics.csv accumulates across all sessions in this hset — plot it to see your progress",
    'use `ft4 generate <model.py> --prompt "..."` to sample from this checkpoint',
]


# ── selector ──────────────────────────────────────────────────────────────


def pick_tip(ctx: TipContext) -> str:
    """Return one tip line. Always returns a non-empty string."""
    rng = random.Random(_seed(ctx))
    for tier in (_TIER1, _TIER2, _TIER3):
        matches = [msg for pred, msg in tier if pred(ctx)]
        if matches:
            return rng.choice(matches)
    return rng.choice(_TIER4)


def _seed(ctx: TipContext) -> int:
    """Stable per-(version, hset, n_sessions) seed.

    Uses sha256 rather than Python's `hash()` because the latter is salted
    per-process (`PYTHONHASHSEED`), which would break test determinism.
    """
    key = f"{ctx.version.name}/{ctx.hset.name}/{ctx.n_sessions}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
