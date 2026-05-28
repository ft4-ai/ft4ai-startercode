"""Unit tests for the briefing tip selector."""
from datetime import datetime, timezone
from pathlib import Path

from mlops.briefing_display import TrainerSummary
from mlops.briefing_tips import TipContext, pick_tip
from mlops.hset_state import Ft4Args, HsetInfo, VersionInfo


# ── helpers ───────────────────────────────────────────────────────────────


def _args(**overrides) -> Ft4Args:
    kwargs = dict(subcommand="train", model_file=Path("model.py"))
    kwargs.update(overrides)
    return Ft4Args(**kwargs)


def _version(name="v001", description=None, dir=None) -> VersionInfo:
    return VersionInfo(
        name=name,
        dir=dir if dir is not None else Path("/tmp"),
        state_dict_hash="a" * 64,
        model_file_hash="b" * 64,
        description=description,
    )


def _hset(name="h001", description=None, dir=None) -> HsetInfo:
    return HsetInfo(
        name=name,
        dir=dir if dir is not None else Path("/tmp"),
        config_hash="c" * 64,
        description=description,
    )


def _ctx(**overrides):
    base = dict(
        args=_args(compile=True, wandb=True),  # silence tier-3 by default
        version=_version(),
        hset=_hset(),
        n_sessions=1,
        last_session={"status": "completed"},
    )
    base.update(overrides)
    return TipContext(**base)


# ── Tier 1 — critical warnings ────────────────────────────────────────────


def test_tier1_error_status_fires_first():
    """Pin `now` close to ended_at so only the error predicate matches."""
    now = datetime(2026, 5, 11, 10, 0, 0, tzinfo=timezone.utc)
    last = {"status": "error", "ended_at": "2026-05-10T10:00:00+00:00"}  # 1 d
    out = pick_tip(_ctx(last_session=last, now=now))
    assert "ended in error" in out


def test_tier1_stale_resume_fires():
    now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    last = {"status": "completed", "ended_at": "2026-05-01T12:00:00+00:00"}  # 19 d
    out = pick_tip(_ctx(last_session=last, now=now))
    assert "over a week ago" in out


def test_tier1_does_not_fire_on_recent_completed_session():
    now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    last = {"status": "completed", "ended_at": "2026-05-19T12:00:00+00:00"}  # 1 d
    out = pick_tip(_ctx(last_session=last, now=now))
    assert "ended in error" not in out
    assert "over a week ago" not in out


# ── Tier 2 — setup nudges ─────────────────────────────────────────────────


def test_tier2_fresh_hset_without_hdesc():
    """Fresh hset + no --hdesc → tier 2 fires; pin vdesc on so only hdesc matches."""
    out = pick_tip(_ctx(
        args=_args(compile=True, wandb=True, vdesc="x"),
        version=_version(description="x"),
        n_sessions=0,
        last_session=None,
    ))
    assert "--hdesc" in out


def test_tier2_skipped_when_hdesc_already_set():
    """User passed --hdesc → tip suppressed; falls through to lower tiers."""
    out = pick_tip(_ctx(
        args=_args(hdesc="my experiment", compile=True, wandb=True),
        n_sessions=0,
        last_session=None,
    ))
    assert "--hdesc" not in out
    assert "--vdesc" in out  # next tier-2 predicate fires (no vdesc set)


def test_tier2_vdesc_fires_when_no_description_and_no_flag():
    out = pick_tip(_ctx(
        args=_args(hdesc="x", compile=True, wandb=True),  # silence tier-2 hdesc + tier-3
        version=_version(description=None),
        n_sessions=0,
        last_session=None,
    ))
    assert "--vdesc" in out


# ── Tier 3 — teaching nudges ──────────────────────────────────────────────


def _make_sibling(tmp_path):
    """Create a sibling hset so the 'only one hset' predicate stays off."""
    sib = tmp_path / "h002"
    sib.mkdir()
    (sib / "hset.json").write_text('{"config_hash": "x"}')


def test_tier3_compile_fires_when_not_set(tmp_path):
    """First predicate in tier-3: --compile not set, all others silenced."""
    _make_sibling(tmp_path)
    out = pick_tip(_ctx(
        args=_args(compile=False, wandb=True, hdesc="x", vdesc="x", prompt=["p"]),
        version=_version(dir=tmp_path, description="x"),
        hset=_hset(dir=tmp_path / "h001"),
        n_sessions=5,
        last_session={"status": "completed"},
        trainer_summary=TrainerSummary(val_check_interval=0.5, gradient_clip_val=1.0),
    ))
    assert "--compile" in out


def test_tier3_wandb_fires_when_only_compile_set(tmp_path):
    """User passed --compile but not --wandb → tier-3 wandb tip."""
    _make_sibling(tmp_path)
    out = pick_tip(_ctx(
        args=_args(compile=True, wandb=False, hdesc="x", vdesc="x", prompt=["p"]),
        version=_version(dir=tmp_path, description="x"),
        hset=_hset(dir=tmp_path / "h001"),
        n_sessions=5,
        last_session={"status": "completed"},
        trainer_summary=TrainerSummary(val_check_interval=0.5, gradient_clip_val=1.0),
    ))
    assert "--wandb" in out


def test_tier3_val_check_interval_fires_at_default(tmp_path):
    """Silence all other tier-3 predicates by passing a sibling hset and
    setting gradient_clip_val; only val_check_interval should fire."""
    sib = tmp_path / "h002"
    sib.mkdir()
    (sib / "hset.json").write_text('{"config_hash": "x"}')
    out = pick_tip(_ctx(
        args=_args(compile=True, wandb=True, hdesc="x", vdesc="x"),
        version=_version(dir=tmp_path, description="x"),
        hset=_hset(dir=tmp_path / "h001"),
        n_sessions=5,
        last_session={"status": "completed"},
        trainer_summary=TrainerSummary(
            val_check_interval=1.0, gradient_clip_val=1.0,
        ),
    ))
    assert "--trainer.val_check_interval" in out


# ── Tier 4 — pedagogical fallback ─────────────────────────────────────────


def test_tier4_fires_when_no_predicate_matches(tmp_path):
    """Everything 'used' → tier 4 must still produce a non-empty tip."""
    sib = tmp_path / "h002"
    sib.mkdir()
    (sib / "hset.json").write_text('{"config_hash": "x"}')
    out = pick_tip(_ctx(
        args=_args(
            compile=True, wandb=True, hdesc="x", vdesc="x",
            prompt=["hello"],
        ),
        version=_version(dir=tmp_path, description="x"),
        hset=_hset(dir=tmp_path / "h001"),
        n_sessions=5,
        last_session={"status": "completed"},
        trainer_summary=TrainerSummary(
            val_check_interval=0.25, gradient_clip_val=1.0,
        ),
    ))
    # Must hit one of the tier-4 student-MLops fallbacks
    assert out  # non-empty
    fallbacks_substrings = [
        "ft4 list", "ft4 show", "ft4 generate",
        "max_epochs", "metrics.csv", "Ctrl-C", "forks",
    ]
    assert any(s in out for s in fallbacks_substrings), \
        f"expected a tier-4 fallback, got: {out!r}"


# ── determinism + rotation ────────────────────────────────────────────────


def test_pick_tip_deterministic_for_same_state():
    """Same (version, hset, n_sessions) always picks the same tip."""
    ctx = _ctx(args=_args(), n_sessions=3, last_session=None)
    a = pick_tip(ctx)
    b = pick_tip(ctx)
    assert a == b


def test_pick_tip_rotates_with_session_count(tmp_path):
    """Tier-4 fallback varies as session count grows → rotation works."""
    base = dict(
        args=_args(
            compile=True, wandb=True, hdesc="x", vdesc="x",
            prompt=["p"],
        ),
        version=_version(dir=tmp_path, description="x"),
        hset=_hset(dir=tmp_path / "h001"),
        last_session={"status": "completed"},
        trainer_summary=TrainerSummary(val_check_interval=0.25, gradient_clip_val=1.0),
    )
    # Create at least one sibling so the "only one hset" predicate doesn't fire.
    sib = tmp_path / "h002"
    sib.mkdir()
    (sib / "hset.json").write_text('{"config_hash": "x"}')

    seen = set()
    for n in range(20):
        seen.add(pick_tip(TipContext(n_sessions=n, **base)))
    # Tier-4 has 4 entries; over 20 session counts we should see >= 2 distinct
    # tips (random.choice on a 4-item list, deterministic per seed).
    assert len(seen) >= 2
