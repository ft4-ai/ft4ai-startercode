"""Unit tests for briefing_display."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.console import Console

from mlops.briefing_display import (
    TrainerSummary,
    _hset_diff_text,
    _humanize_age,
    _humanize_seconds,
    _pace_text,
    _plan_text,
    render_briefing,
    render_briefing_plain,
)
from mlops.hset_state import Ft4Args, HsetInfo, VersionInfo


def _args(**overrides) -> Ft4Args:
    """Default Ft4Args for tip-predicate tests."""
    kwargs = dict(subcommand="train", model_file=Path("model.py"))
    kwargs.update(overrides)
    return Ft4Args(**kwargs)


# ── Helpers ────────────────────────────────────────────────────────────────

def _version(name="v001", description=None, created_at=None) -> VersionInfo:
    return VersionInfo(
        name=name,
        dir=Path("/tmp"),
        state_dict_hash="a" * 64,
        model_file_hash="b" * 64,
        description=description,
        created_at=created_at,
    )


def _hset(name="h001", description=None, dir: Path | None = None) -> HsetInfo:
    return HsetInfo(
        name=name,
        dir=dir if dir is not None else Path("/tmp"),
        config_hash="c" * 64,
        description=description,
    )


def _render_to_text(*args, width: int = 200, **kwargs) -> str:
    """Render the briefing panel to plain text for substring assertions."""
    panel = render_briefing(*args, **kwargs)
    console = Console(record=True, force_terminal=True, width=width, color_system=None)
    console.print(panel)
    return console.export_text()


def _session(
    idx=1,
    started_at=None,
    ended_at="2026-05-09T10:00:00+00:00",
    train_key="train_ce_epoch",
    train_value=2.5,
    val_key="val_ce_epoch",
    val_value=2.8,
    epochs=2,
    steps=None,
    est_sec_per_epoch=None,
) -> dict:
    rec: dict = {
        "session_index": idx,
        "ended_at": ended_at,
        "epochs": epochs,
    }
    if started_at is not None:
        rec["started_at"] = started_at
    if steps is not None:
        rec["steps"] = steps
    if train_value is not None:
        rec["train_metric_key"] = train_key
        rec["train_metric_value"] = train_value
    if val_value is not None:
        rec["val_metric_key"] = val_key
        rec["val_metric_value"] = val_value
    if est_sec_per_epoch is not None:
        rec["est_sec_per_epoch"] = est_sec_per_epoch
    return rec


# ── fresh vs resume signal (title verb + border color) ──────────────────


def _render_panel(*args, **kwargs):
    """Return the Panel object (for inspecting border_style etc.)."""
    return render_briefing(*args, **kwargs)


def test_fresh_title_says_train_from_scratch():
    out = _render_to_text(Path("iris.py"), _version(), _hset(), [], ckpt_path=None)
    assert "ft4 train from scratch:" in out


def test_resume_title_says_resume_training(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), [_session()], ckpt_path=ckpt,
    )
    assert "ft4 resume training:" in out


def test_fresh_border_is_blue():
    p = _render_panel(Path("iris.py"), _version(), _hset(), [], ckpt_path=None)
    assert p.border_style == "blue"


def test_resume_border_is_green(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    p = _render_panel(
        Path("iris.py"), _version(), _hset(), [_session()], ckpt_path=ckpt,
    )
    assert p.border_style == "green"


def test_backtrack_border_is_magenta(tmp_path):
    """Explicit backtrack mode → magenta border (the eye-catching colour
    now flags the genuinely-unusual case)."""
    from mlops.hset_state import Mode
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    p = _render_panel(
        Path("iris.py"), _version(), _hset(), [_session()], ckpt_path=ckpt,
        mode=Mode.BACKTRACK,
    )
    assert p.border_style == "magenta"


def test_backtrack_title_verb(tmp_path):
    """Backtrack mode → 'backtrack and resume training'."""
    from mlops.hset_state import Mode
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), [_session()], ckpt_path=ckpt,
        mode=Mode.BACKTRACK,
    )
    assert "ft4 backtrack and resume training:" in out


def test_plain_title_carries_verb_fresh():
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(), sessions=[], ckpt_path=None,
    )
    assert out.splitlines()[0].startswith("ft4 train from scratch:")


def test_plain_title_carries_verb_resume(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(),
        sessions=[_session()], ckpt_path=ckpt,
    )
    assert out.splitlines()[0].startswith("ft4 resume training:")


# ── version row: 'modified' suffix, no hash ──────────────────────────────


def test_version_row_drops_hash():
    """The 8-char hash is no longer displayed on the version row."""
    out = _render_to_text(Path("iris.py"), _version(), _hset(), [], None)
    # First 8 of state_dict_hash "aaaaaaaa" must not appear in the briefing.
    assert "aaaaaaaa" not in out


def test_version_row_shows_modified_for_existing_version():
    """Loaded version: 'modified N d ago' suffix from created_at."""
    now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    v = _version(created_at="2026-05-09T10:00:00+00:00")  # 2 d ago
    out = _render_to_text(Path("iris.py"), v, _hset(), [], None, now=now)
    assert "modified 2 d ago" in out


def test_version_row_omits_modified_when_just_created():
    """Fresh-this-run version (created < 60s ago): no time suffix."""
    now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    v = _version(created_at="2026-05-11T11:59:55+00:00")  # 5s ago
    out = _render_to_text(Path("iris.py"), v, _hset(), [], None, now=now)
    assert "modified" not in out


def test_version_row_omits_modified_when_created_at_missing():
    """Older versions without created_at: no suffix, no crash."""
    out = _render_to_text(
        Path("iris.py"), _version(created_at=None), _hset(), [], None,
    )
    assert "modified" not in out


# ── render_briefing: new hset ─────────────────────────────────────────────

def test_new_hset_includes_NEW_label():
    out = _render_to_text(Path("iris.py"), _version(), _hset(),
                         sessions=[], ckpt_path=None)
    assert "NEW" in out


def test_new_hset_includes_all_three_ids():
    out = _render_to_text(Path("iris.py"), _version("v007"), _hset("h042"), [], None)
    assert "iris.py" in out
    assert "v007" in out
    assert "h042" in out


def test_new_hset_no_ckpt_says_from_scratch():
    out = _render_to_text(Path("iris.py"), _version(), _hset(), [], ckpt_path=None)
    assert "from scratch" in out


# ── row order ─────────────────────────────────────────────────────────────

def test_row_order_is_model_version_hset_ckpt():
    """Row order matches the student's mental model. The metrics + perf
    lines come AFTER ckpt now (they characterize the ckpt)."""
    out = _render_to_text(Path("iris.py"), _version(), _hset(), [], None)
    labels_in_order = ["model", "version", "hset", "ckpt"]
    indices = [out.find(label) for label in labels_in_order]
    assert all(i > 0 for i in indices), f"missing labels: {indices}"
    assert indices == sorted(indices), f"labels out of order: {indices}"


def test_metrics_render_inline_with_ckpt_row(tmp_path):
    """Metrics ride along on the `ckpt` row line itself — no standalone
    `metrics` row. The label `ckpt` and the value `val_ce=…` appear in
    sequence with no other labeled row between them."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x" * 2048)  # so the size renders
    sessions = [_session(steps=120, est_sec_per_epoch=10.5)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), sessions, ckpt_path=ckpt,
    )
    ckpt_idx = out.find("ckpt")
    val_idx = out.find("val_ce=")
    assert 0 < ckpt_idx < val_idx
    # No standalone metrics row — the label "metrics" should not appear
    # anywhere on a row of its own.
    assert "\nmetrics" not in out and "  metrics" not in out


# ── existing hset with sessions ───────────────────────────────────────────

def test_existing_hset_session_count_singular():
    sessions = [_session(1)]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None)
    assert "1 session" in out
    assert "1 sessions" not in out


def test_existing_hset_session_count_plural():
    sessions = [_session(1), _session(2), _session(3)]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None)
    assert "3 sessions" in out


def test_existing_hset_renders_metric_names_and_values(tmp_path):
    """The metric *key* is shown, e.g. `train_ce=1.820`, not `train=1.820`.
    Metrics ride along on the ckpt row, so the test needs a ckpt loaded."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x" * 1024)
    sessions = [_session(1, train_value=1.82, val_value=2.31)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), sessions, ckpt_path=ckpt,
    )
    assert "train_ce=1.820" in out
    assert "val_ce=2.310" in out


def test_step_metric_key_kept_in_display(tmp_path):
    """If we only had a _step value (no _epoch), keep the suffix so the
    student knows it's a single-batch value."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x" * 1024)
    sessions = [_session(1, train_key="train_ce_step", train_value=1.5,
                         val_key="val_ce_step", val_value=1.7)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), sessions, ckpt_path=ckpt,
    )
    assert "train_ce_step=1.500" in out
    assert "val_ce_step=1.700" in out


def test_existing_hset_renders_ckpt_name():
    sessions = [_session(1)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), sessions,
        ckpt_path=Path("/some/path/last.ckpt"),
    )
    assert "last.ckpt" in out
    # Full path should not appear (just the basename).
    assert "/some/path/last.ckpt" not in out


def test_existing_hset_humanizes_session_age():
    now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    sessions = [_session(1, ended_at="2026-05-09T10:00:00+00:00")]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None,
                         now=now)
    assert "2 d ago" in out


# ── sec/epoch (from est_sec_per_epoch on the session record) ─────────────

def test_speed_rendered_when_record_has_est_sec_per_epoch():
    """The briefing reads `est_sec_per_epoch` (stored by SessionFinalizationCallback),
    not by recomputing from wall-time."""
    sessions = [_session(1, steps=200, est_sec_per_epoch=10.5)]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None)
    assert "sec/epoch" in out
    assert "10.5" in out


# ── descriptions ──────────────────────────────────────────────────────────

def test_no_descriptions_no_none_string():
    out = _render_to_text(Path("iris.py"), _version(), _hset(), [], None)
    assert "None" not in out


def test_version_description_rendered():
    v = _version(description="bigger model")
    out = _render_to_text(Path("iris.py"), v, _hset(), [], None)
    assert "bigger model" in out


def test_hset_description_rendered():
    t = _hset(description="lr warmup tweak")
    out = _render_to_text(Path("iris.py"), _version(), t, [], None)
    assert "lr warmup tweak" in out


# ── robustness ────────────────────────────────────────────────────────────

def test_session_with_only_partial_fields_renders():
    """Caller's sessions.jsonl shape may evolve; render shouldn't crash on
    missing fields."""
    sessions = [{"session_index": 1}]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None)
    assert "1 session" in out


def test_session_with_bogus_ended_at_does_not_crash():
    sessions = [{"session_index": 1, "ended_at": "not-an-iso-timestamp"}]
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None)
    assert "1 session" in out


def test_session_with_no_metric_fields_renders_no_metric_line():
    sessions = [{
        "session_index": 1,
        "ended_at": "2026-05-09T10:00:00+00:00",
    }]
    now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    out = _render_to_text(Path("iris.py"), _version(), _hset(), sessions, None, now=now)
    assert "train_" not in out
    assert "val_" not in out


# ── symlink rendering in the paths tree ──────────────────────────────────

def test_path_tree_resolves_last_ckpt_symlink_in_rich_renderer(tmp_path):
    """When `last.ckpt` is a true symlink, the paths tree shows
    `-> step=NNNNN.ckpt`. (The ckpt row itself doesn't duplicate this.)"""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    target = ckpt_dir / "step=00500.ckpt"
    target.write_bytes(b"")
    last = ckpt_dir / "last.ckpt"
    last.symlink_to(target.name)

    sessions = [_session(1)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), sessions, ckpt_path=last,
    )
    assert "last.ckpt" in out
    assert "step=00500.ckpt" in out
    assert "->" in out  # ASCII arrow (Unicode → glitches on some terminals)


def test_ckpt_row_handles_plain_file(tmp_path):
    """Non-symlink ckpt: no arrow, no target — just the name."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"")

    sessions = [_session(1)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), sessions, ckpt_path=ckpt,
    )
    assert "last.ckpt" in out
    assert "->" not in out


# ── paths tree contents ───────────────────────────────────────────────────

def test_paths_tree_includes_config_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("foo: bar")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "config.yaml" in out


def test_paths_tree_excludes_sessions_jsonl(tmp_path):
    """sessions.jsonl was deliberately dropped from the paths tree."""
    (tmp_path / "sessions.jsonl").write_text("")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "sessions.jsonl" not in out


def test_paths_tree_expands_latest_session(tmp_path):
    """The latest sessions/sNNN/ subtree is expanded to surface the two
    artifacts students care about: hparams.yaml and runconfig.yaml."""
    s2 = tmp_path / "sessions" / "s002"
    s2.mkdir(parents=True)
    (s2 / "hparams.yaml").write_text("foo: 1\n")
    (s2 / "runconfig.yaml").write_text("bar: 2\n")
    (tmp_path / "sessions" / "s001").mkdir(parents=True)
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "sessions/s002/" in out
    assert "hparams.yaml" in out
    assert "runconfig.yaml" in out
    assert "last of 2" in out  # latest-of-N annotation


def test_paths_tree_shows_params_at_step(tmp_path):
    """The ckpt entry in the paths tree shows the step number from the file
    on disk — not from the session record, which can be ahead of the last
    successful save (e.g. session errored)."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step_000500.ckpt").write_bytes(b"")
    (tmp_path / "checkpoints" / "last.ckpt").symlink_to("step_000500.ckpt")
    sessions = [_session(1, steps=500)]
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), sessions, None,
    )
    assert "params @ step 500" in out


def test_path_tree_infers_last_ckpt_target_from_step_files(tmp_path):
    """Lightning's save_last writes last.ckpt as a regular file copy (not
    a symlink). The paths tree should still show `-> step=N.ckpt` by
    inferring from the highest-numbered step ckpt in the same dir."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "last.ckpt").write_bytes(b"x")  # regular file, NOT a symlink
    (ckpts / "step=42.ckpt").write_bytes(b"x")
    (ckpts / "step=12.ckpt").write_bytes(b"x")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [_session(1, steps=42)], None,
    )
    assert "-> step=42.ckpt" in out


def test_ckpt_row_omits_target_continuation_line(tmp_path):
    """The `ckpt` row never carries a `-> step=N.ckpt` continuation line.
    The paths tree owns that display — duplicating it would be noise."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    last = ckpts / "last.ckpt"
    last.write_bytes(b"x")
    (ckpts / "step=99.ckpt").write_bytes(b"x")
    out = render_briefing_plain(
        Path("iris.py"), _version(), _hset(dir=tmp_path),
        sessions=[_session(1, steps=99)], ckpt_path=last,
    )
    # Path tree still shows the target (covered by another test); the
    # ckpt row itself does not add its own redundant continuation.
    # Verify by checking no line that starts with whitespace + '-> ' appears
    # directly under the `ckpt:` label.
    rows = out.splitlines()
    ckpt_row_idx = next(i for i, line in enumerate(rows) if "ckpt:" in line)
    next_row = rows[ckpt_row_idx + 1] if ckpt_row_idx + 1 < len(rows) else ""
    assert "->" not in next_row, f"unexpected continuation under ckpt: {next_row!r}"


def test_paths_tree_shows_latest_sample(tmp_path):
    """When samples/ has files, the LATEST one is named in the tree."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "step_000050.md").write_text("")
    (samples / "step_000500.md").write_text("")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "samples/step_000500.md" in out


# ── plan row ──────────────────────────────────────────────────────────────


def test_plan_text_fresh_epochs():
    t = TrainerSummary(max_epochs=100)
    assert _plan_text(t, last_session=None) == "train 100 epochs"


def test_plan_text_fresh_single_epoch():
    t = TrainerSummary(max_epochs=1)
    assert _plan_text(t, last_session=None) == "train 1 epoch"


def test_plan_text_resume_with_known_epochs():
    t = TrainerSummary(max_epochs=150)
    last = {"epochs": 100}
    assert _plan_text(t, last_session=last) == "add 50 epochs (total -> 150)"


def test_plan_text_resume_with_missing_epochs():
    t = TrainerSummary(max_epochs=100)
    last = {"steps": 5000}  # no 'epochs' field
    assert _plan_text(t, last_session=last) == "add up to 100 epochs (total -> 100)"


def test_plan_text_step_based_fresh():
    t = TrainerSummary(max_steps=10000)
    assert _plan_text(t, last_session=None) == "train up to 10,000 steps"


def test_plan_text_step_based_resume():
    t = TrainerSummary(max_steps=20000)
    assert _plan_text(t, last_session={"steps": 5000}) == "add up to 20,000 steps"


def test_plan_text_no_trainer_returns_empty():
    assert _plan_text(None, last_session=None) == ""


def test_plan_text_no_stopping_criterion_returns_empty():
    """No max_epochs or max_steps set → no plan row."""
    t = TrainerSummary()
    assert _plan_text(t, last_session=None) == ""


def test_plan_row_renders_in_briefing(tmp_path):
    """End-to-end: plan row appears between ckpt and data."""
    t = TrainerSummary(max_epochs=50)
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), [], None,
        trainer=t,
    )
    assert "plan" in out
    assert "train 50 epochs" in out


# ── hset diff line ────────────────────────────────────────────────────────


def _setup_sibling_hset(tmp_path, name="h001", config_yaml=""):
    """Create a sibling hset directory with hset.json and config.yaml."""
    d = tmp_path / name
    d.mkdir()
    (d / "hset.json").write_text(f'{{"config_hash": "{name}-hash"}}')
    (d / "config.yaml").write_text(config_yaml)
    return d


def test_hset_diff_no_config_returns_empty():
    assert _hset_diff_text(_version(), _hset(), config=None) == ""


def test_hset_diff_no_sibling_returns_empty(tmp_path):
    """Only one hset in the version → no diff line."""
    v = VersionInfo(name="v001", dir=tmp_path,
                    state_dict_hash="a"*64, model_file_hash="b"*64)
    h = HsetInfo(name="h001", dir=tmp_path / "h001", config_hash="x")
    config = {"model": {"lr": 1e-4}}
    assert _hset_diff_text(v, h, config=config) == ""


def test_hset_diff_renders_changed_keys(tmp_path):
    # LR and dropout are runconfig keys; scientific (hset) keys are pure
    # architecture/data shape (dim, depth, seq_len, dataset).
    _setup_sibling_hset(tmp_path, "h001",
                        "model:\n  dim: 256\n  depth: 4\n")
    v = VersionInfo(name="v001", dir=tmp_path,
                    state_dict_hash="a"*64, model_file_hash="b"*64)
    h = HsetInfo(name="h002", dir=tmp_path / "h002", config_hash="x")
    config = {"model": {"dim": 512, "depth": 8}}
    out = _hset_diff_text(v, h, config=config)
    assert out.startswith("vs h001:")
    assert "model.dim" in out
    assert "model.depth" in out
    # Integer formatting via :g
    assert "256" in out or "512" in out


def test_hset_diff_truncates_after_5(tmp_path):
    prior = "\n".join(f"k{i}: {i}" for i in range(10))
    _setup_sibling_hset(tmp_path, "h001", prior)
    v = VersionInfo(name="v001", dir=tmp_path,
                    state_dict_hash="a"*64, model_file_hash="b"*64)
    h = HsetInfo(name="h002", dir=tmp_path / "h002", config_hash="x")
    # All keys differ — 10 diffs, expect "+5 more"
    config = {f"k{i}": 100 + i for i in range(10)}
    out = _hset_diff_text(v, h, config=config)
    assert "+5 more" in out


def test_hset_diff_excludes_trainer_loggers(tmp_path):
    """Excluded prefixes (loggers, callbacks, LR/optimizer, etc.) shouldn't
    appear in the hset diff — only scientifically meaningful keys."""
    _setup_sibling_hset(tmp_path, "h001",
                        "trainer:\n  logger: false\nmodel:\n  dim: 256\n")
    v = VersionInfo(name="v001", dir=tmp_path,
                    state_dict_hash="a"*64, model_file_hash="b"*64)
    h = HsetInfo(name="h002", dir=tmp_path / "h002", config_hash="x")
    config = {"trainer": {"logger": True}, "model": {"dim": 512}}
    out = _hset_diff_text(v, h, config=config)
    assert "trainer.logger" not in out  # excluded
    assert "model.dim" in out


def test_hset_diff_skips_when_sibling_config_missing(tmp_path):
    """Sibling exists but has no config.yaml → no diff line, no crash."""
    d = tmp_path / "h001"
    d.mkdir()
    (d / "hset.json").write_text('{"config_hash": "x"}')
    v = VersionInfo(name="v001", dir=tmp_path,
                    state_dict_hash="a"*64, model_file_hash="b"*64)
    h = HsetInfo(name="h002", dir=tmp_path / "h002", config_hash="y")
    assert _hset_diff_text(v, h, config={"model": {"lr": 1.0}}) == ""


# ── pace row ──────────────────────────────────────────────────────────────


def test_pace_text_renders_per_epoch_only():
    """Pace shows just the per-epoch rate now; total-time projection lives
    in the plan row instead."""
    out = _pace_text((360.0, "(recent)"))
    assert "6 min" in out and "/epoch" in out
    assert "(recent)" in out
    assert "total" not in out
    assert "to finish" not in out


def test_pace_text_none_returns_placeholder():
    """No estimate yet → placeholder."""
    assert _pace_text(None) == "measured after epoch 1"


def test_pace_text_renders_sibling_source():
    """Sibling-sourced estimates carry a '(based on hNNN)' label."""
    out = _pace_text((600.0, "(based on h001)"))
    assert "10 min" in out and "/epoch" in out
    assert "(based on h001)" in out


def test_plan_text_includes_projected_total_time():
    """The total-time projection moved from pace to plan: when sec_per_epoch
    is known, plan appends '≈ Th Mmin' over the remaining epochs."""
    t = TrainerSummary(max_epochs=150)
    last = {"epochs": 50}
    # 100 remaining epochs × 360s = 36000s = 10 h
    out = _plan_text(t, last_session=last, sec_per_epoch=360.0)
    assert "add 100 epochs (total -> 150)" in out
    assert "10 h" in out


def test_pace_row_renders_in_briefing(tmp_path):
    last = _session(est_sec_per_epoch=120.0, epochs=5)
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), [last], None,
        trainer=TrainerSummary(max_epochs=10),
    )
    assert "pace" in out
    assert "/epoch" in out


def test_humanize_seconds_days():
    """≥ 24 h should render as 'd h' not '48 h 00 min'."""
    assert _humanize_seconds(48 * 3600) == "2 d 00 h"
    assert _humanize_seconds(86400 + 3600) == "1 d 01 h"


def test_humanize_seconds_just_under_day():
    """23 h 59 min stays in the hours branch."""
    assert _humanize_seconds(86399).startswith("23 h")


# ── path tree: empty-state entries ───────────────────────────────────────


def test_paths_tree_empty_checkpoints_dir_shows_will_fill(tmp_path):
    """Fresh hset: checkpoints/ exists but is empty → show empty-state row."""
    (tmp_path / "checkpoints").mkdir()
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "checkpoints/" in out
    assert "empty" in out
    assert "will fill" in out


def test_paths_tree_empty_samples_shown_for_generative(tmp_path):
    """Fresh generative model: empty samples/ shows up so the user knows where
    output will land."""
    class _Gen:
        def generate_samples(self): pass
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
        pl_module=_Gen(),
    )
    assert "samples/" in out


def test_paths_tree_empty_samples_hidden_for_nongenerative(tmp_path):
    """Non-generative model: no samples/ row when empty."""
    class _NonGen:
        pass
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
        pl_module=_NonGen(),
    )
    # samples/ entry shouldn't appear since dir doesn't exist and model
    # isn't generative.
    assert "samples/" not in out


# ── param count ───────────────────────────────────────────────────────────

class _FakeModule:
    """Stand-in for a Lightning module with a known parameter count."""
    class _Param:
        def __init__(self, n): self._n = n
        def numel(self): return self._n

    def __init__(self, total: int):
        self._params = [self._Param(total)]

    def parameters(self):
        return iter(self._params)


def test_param_count_rendered_when_pl_module_provided():
    fake = _FakeModule(total=4_200_000)
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(), [], None, pl_module=fake,
    )
    assert "4.2 M params" in out


# ── _humanize_age ─────────────────────────────────────────────────────────

@pytest.fixture
def now():
    return datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)


def test_humanize_just_now(now):
    assert _humanize_age(now - timedelta(seconds=15), now=now) == "just now"


def test_humanize_minutes(now):
    assert _humanize_age(now - timedelta(minutes=5), now=now) == "5 min ago"


def test_humanize_hours(now):
    assert _humanize_age(now - timedelta(hours=3), now=now) == "3 h ago"


def test_humanize_days(now):
    assert _humanize_age(now - timedelta(days=2), now=now) == "2 d ago"


def test_humanize_negative_delta_treated_as_now(now):
    # Future timestamp shouldn't crash or produce negative output.
    assert _humanize_age(now + timedelta(minutes=5), now=now) == "just now"


def test_humanize_naive_datetime_treated_as_utc(now):
    # Defensive: a stored timestamp that lost its tz still works.
    naive = (now - timedelta(hours=4)).replace(tzinfo=None)
    assert _humanize_age(naive, now=now) == "4 h ago"


# ── tips (rendering integration; see test_briefing_tips.py for selector) ──


def test_fresh_hset_with_no_hdesc_shows_hdesc_tip():
    """Fresh hset, no --hdesc passed → tier-2 hdesc tip fires.

    We also pass `vdesc` so the other tier-2 predicate is suppressed and
    only the hdesc one matches — within-tier selection is random, so
    asserting the specific tip needs the other to be off.
    """
    out = _render_to_text(
        Path("model.py"), _version(description="x"), _hset(),
        sessions=[], ckpt_path=None,
        args=_args(vdesc="x"),
    )
    assert "--hdesc" in out


def test_tip_row_always_present_in_briefing():
    """No matter the state, the tip row should appear (tier 4 fallback)."""
    out = _render_to_text(
        Path("model.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        args=_args(),
    )
    assert "tip" in out.lower()


# ── render_briefing_plain ─────────────────────────────────────────────────

def test_plain_contains_core_facts(tmp_path):
    """Title includes the next session id (`· sNNN`) and the basic rows
    appear. Inline ckpt metrics are tested separately — they require a
    ckpt path to render."""
    h = _hset(dir=tmp_path)
    (tmp_path / "config.yaml").write_text("model: foo\n")
    out = render_briefing_plain(
        Path("model.py"), _version(name="v007"), h,
        sessions=[_session(steps=1000, started_at="2026-05-09T09:00:00+00:00")],
        ckpt_path=None,
    )
    # Title with next-session suffix (1 prior session → next is s002).
    assert "ft4 train from scratch: model.py · v007 · h001 · s002" in out
    assert "model:" in out
    assert "version:" in out
    assert "v007" in out
    assert "hset:" in out
    assert "h001" in out
    assert "ckpt:" in out
    assert "paths:" in out
    assert "config.yaml" in out


def test_plain_ckpt_inline_metrics_when_ckpt_present(tmp_path):
    """Metrics ride along on the ckpt row, not as a standalone row."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x" * 2048)
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(dir=tmp_path),
        sessions=[_session(steps=1000)],
        ckpt_path=ckpt,
    )
    assert "train_ce=2.500" in out
    assert "val_ce=2.800" in out
    assert "2 KB" in out  # 2048-byte ckpt → 2 KB (1024-base)


def test_plain_has_no_panel_borders(tmp_path):
    """No box-drawing characters, no ANSI escapes — that's the whole point."""
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(dir=tmp_path),
        sessions=[], ckpt_path=None,
    )
    for c in "┌─└│├┐┘┤┬┴┼":
        assert c not in out, f"plain text contained box-drawing char: {c!r}"
    # No ANSI escapes.
    assert "\x1b[" not in out


def test_plain_includes_tip_for_fresh_hset():
    out = render_briefing_plain(
        Path("model.py"), _version(description="x"), _hset(),
        sessions=[], ckpt_path=None,
        args=_args(vdesc="x"),
    )
    assert "tip:" in out
    assert "--hdesc" in out


def test_plain_always_includes_tip_row(tmp_path):
    """Tier-4 fallback guarantees the tip row appears."""
    h = _hset(dir=tmp_path)
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "step_000010.md").write_text("# sample")
    out = render_briefing_plain(
        Path("model.py"), _version(), h,
        sessions=[_session()], ckpt_path=None,
        args=_args(),
    )
    assert "tip:" in out


def test_plain_path_tree_resolves_last_ckpt_symlink(tmp_path):
    """When last.ckpt IS a real symlink (older Lightning behavior), the
    path tree shows '-> <target>' alongside it. (The ckpt row itself
    doesn't duplicate this — see test_ckpt_row_omits_target_continuation_line.)"""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    target = ckpts / "step=42.ckpt"
    target.write_text("")
    last = ckpts / "last.ckpt"
    last.symlink_to(target.name)
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(dir=tmp_path),
        sessions=[_session(1, steps=42)], ckpt_path=last,
    )
    # The path tree row for last.ckpt carries the symlink target.
    assert "checkpoints/last.ckpt" in out
    assert "-> step=42.ckpt" in out


def test_plain_path_tree_lists_every_entry(tmp_path):
    h = _hset(dir=tmp_path)
    (tmp_path / "config.yaml").write_text("")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step_000050.ckpt").write_text("")
    (tmp_path / "checkpoints" / "last.ckpt").symlink_to("step_000050.ckpt")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "step_000050.md").write_text("")
    (tmp_path / "sessions").mkdir()
    s001 = tmp_path / "sessions" / "s001"
    s001.mkdir()
    (s001 / "hparams.yaml").write_text("")
    (s001 / "runconfig.yaml").write_text("")
    (tmp_path / "metrics.csv").write_text("")
    out = render_briefing_plain(
        Path("model.py"), _version(), h,
        sessions=[_session(steps=50)], ckpt_path=None,
    )
    assert "config.yaml" in out
    assert "full config" in out
    assert "checkpoints/last.ckpt" in out
    assert "params @ step 50" in out
    assert "samples/step_000050.md" in out
    assert "latest output" in out
    assert "sessions/s001/" in out
    assert "hparams.yaml" in out
    assert "runconfig.yaml" in out
    assert "metrics.csv" in out
    assert "cumulative" in out


def test_plain_datamodule_summary():
    """Data row: `<Class>  N records  ·  M batches` when batch_size is known,
    else just `<Class>  N records`."""
    class _DM:
        class _DS:
            def __len__(self): return 1234
        train_dataset = _DS()
        batch_size = 100
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        datamodule=_DM(),
    )
    assert "data:" in out
    assert "_DM" in out
    assert "1,234 records" in out
    assert "13 batches" in out  # ceil(1234 / 100)


def test_plain_datamodule_summary_without_batch_size():
    """When batch_size can't be determined, omit the batches half."""
    class _DM:
        class _DS:
            def __len__(self): return 1234
        train_dataset = _DS()
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        datamodule=_DM(),
    )
    assert "1,234 records" in out
    assert "batches" not in out


def test_plain_datamodule_summary_surfaces_init_args():
    """Data row surfaces scientifically-meaningful init_args from config —
    the bare class name leaves "which dataset slice" invisible."""
    class _DM:
        pass
    config = {
        "data": {
            "class_path": "ft4.pipeline.stories_data_module.StoriesDataModule",
            "init_args": {
                "data_size": "full",
                "seq_len": 300,
                "batch_size": 64,        # excluded — covered by "batches" tail
                "num_workers": 8,        # excluded — infra noise
            },
        }
    }
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        datamodule=_DM(), config=config,
    )
    assert "_DM(data_size='full'" in out
    assert "seq_len=300" in out
    assert "batch_size" not in out
    assert "num_workers" not in out


def test_plain_datamodule_summary_degrades_when_no_config():
    """Without config or matching keys, falls back to bare class name."""
    class _DM:
        pass
    out = render_briefing_plain(
        Path("model.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        datamodule=_DM(), config=None,
    )
    # Bare class name, no parentheses.
    assert "_DM" in out
    assert "_DM(" not in out


def test_paths_tree_params_at_step_reads_symlink_not_session(tmp_path):
    """Bug 1 regression: when the session's global_step is ahead of the
    last successful save (session errored), `params @ step N` must reflect
    the file on disk, not the session record."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step_006280.ckpt").write_bytes(b"x")
    (ckpts / "last.ckpt").symlink_to("step_006280.ckpt")
    # session reached step 7826 then errored — last successful save is 6280.
    sessions = [_session(1, steps=7826)]
    out = _render_to_text(
        Path("transformer.py"), _version(), _hset(dir=tmp_path), sessions, None,
    )
    assert "params @ step 6280" in out
    assert "params @ step 7826" not in out


def test_paths_tree_accepts_new_step_underscore_format(tmp_path):
    """Both legacy `step=N.ckpt` and new `step_NNNNNN.ckpt` parse correctly."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step_000123.ckpt").write_bytes(b"x")
    (ckpts / "last.ckpt").symlink_to("step_000123.ckpt")
    out = _render_to_text(
        Path("iris.py"), _version(), _hset(dir=tmp_path), [], None,
    )
    assert "step_000123.ckpt" in out
    assert "params @ step 123" in out


def test_plain_param_count_from_pl_module():
    import torch
    Iris = type("Iris", (), {})
    Iris.__module__ = "iris"
    pl_module = Iris()
    pl_module.parameters = lambda: [torch.zeros(100), torch.zeros(200)]

    out = render_briefing_plain(
        Path("iris.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        pl_module=pl_module,
    )
    assert "iris.Iris" in out
    assert "300 params" in out  # 100 + 200


# ── _humanize_bytes (ckpt-row size formatter) ─────────────────────────────

def test_humanize_bytes_under_1kb():
    from mlops.briefing_display import _humanize_bytes
    assert _humanize_bytes(0) == "0 B"
    assert _humanize_bytes(1023) == "1023 B"


def test_humanize_bytes_kb_mb_gb():
    from mlops.briefing_display import _humanize_bytes
    # Integer-only — these sizes are rounded, not estimated.
    assert _humanize_bytes(2048) == "2 KB"
    assert _humanize_bytes(5 * 1024 * 1024) == "5 MB"
    assert _humanize_bytes(3 * 1024 ** 3) == "3 GB"


# ── samples preview row ───────────────────────────────────────────────────

_SAMPLE_MD = """# Samples @ step 50

**Prompt:** `Once upon a time`

Once upon a time, there was a little girl who loved to play in the garden.

---

**Prompt:** `The dog`

The dog ran across the field and jumped over the fence.

---

**Prompt:** `She opened the door`

She opened the door and saw a beautiful garden filled with flowers.

---
"""


def test_samples_preview_renders_when_files_exist(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "step_000050.md").write_text(_SAMPLE_MD)
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(dir=tmp_path),
        sessions=[], ckpt_path=None,
    )
    assert "samples:" in out
    assert "Once upon a time" in out
    # All three pairs should appear (test markdown has exactly 3).
    assert "The dog" in out
    assert "She opened the door" in out


def test_samples_preview_omitted_when_directory_missing(tmp_path):
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(dir=tmp_path),
        sessions=[], ckpt_path=None,
    )
    assert "samples:" not in out


def test_samples_preview_picks_latest_file(tmp_path):
    """Latest-numbered sample file wins (lexical sort of `step_*.md`)."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "step_000050.md").write_text(
        "**Prompt:** `OLD`\n\nOLD continuation\n\n---\n"
    )
    (samples / "step_000100.md").write_text(
        "**Prompt:** `NEW`\n\nNEW continuation\n\n---\n"
    )
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(dir=tmp_path),
        sessions=[], ckpt_path=None,
    )
    # Render shape: `▸ <prompt><continuation>` — no quotes; the Rich
    # renderer dims the prompt to mark its boundary, plain text just runs
    # them together.
    assert "▸ NEW" in out
    assert "continuation" in out
    assert "OLD" not in out
    # No leftover quoted-prompt syntax.
    assert '"NEW"' not in out


# ── runconfig row ─────────────────────────────────────────────────────────

def test_runconfig_row_renders_rdesc_when_set(tmp_path):
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        args=_args(rdesc="warmup + cosine, lr=3e-4"),
        curr_runconfig={"model.lr": 3e-4, "trainer.precision": "bf16-mixed"},
    )
    assert "runconfig:" in out
    assert '"warmup + cosine, lr=3e-4"' in out


def test_runconfig_row_summary_when_no_rdesc(tmp_path):
    """No rdesc → fall back to a compact summary line of top recipe keys."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={
            "model.lr": 3e-4,
            "model.weight_decay": 1e-2,
            "trainer.precision": "bf16-mixed",
        },
    )
    assert "runconfig:" in out
    assert "model.lr=0.0003" in out
    assert "trainer.precision=" in out


def test_runconfig_row_shows_diff_continuation_when_prev_differs(tmp_path):
    """When prev_runconfig differs from curr, a `vs sNNN: …` line appears."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={"model.lr": 3e-4, "trainer.precision": "bf16-mixed"},
        prev_runconfig={"model.lr": 1e-4, "trainer.precision": "bf16-mixed"},
        prev_session_id="s001",
    )
    assert "vs s001:" in out
    assert "model.lr" in out
    assert "0.0001" in out and "0.0003" in out


def test_runconfig_row_no_diff_continuation_when_prev_matches(tmp_path):
    """Identical prev_runconfig → no `vs sNNN:` continuation line."""
    rc = {"model.lr": 3e-4}
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig=rc,
        prev_runconfig=dict(rc),
        prev_session_id="s001",
    )
    assert "vs s001:" not in out


def test_runconfig_row_omitted_when_no_data():
    """No rdesc, empty runconfig dict, no prev → no row at all."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={},
    )
    assert "runconfig:" not in out


def _diff_line(out: str) -> str:
    """Pull the 'vs sNNN: …' continuation line from a rendered briefing
    (or empty string if there isn't one). Used to assert what's in the diff
    *specifically*, ignoring noise keys that also appear in the summary."""
    for line in out.splitlines():
        if "vs s" in line and ":" in line:
            return line
    return ""


def test_runconfig_diff_filters_seed_everything():
    """seed_everything is auto-randomized per invocation — useless in the
    vs-prev diff. It stays in runconfig.yaml AND the summary line; the
    diff display alone hides it."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={"model.lr": 3e-4, "seed_everything": 12345},
        prev_runconfig={"model.lr": 3e-4, "seed_everything": 67890},
        prev_session_id="s001",
    )
    # Only diff-able key was filtered → no vs line at all.
    assert "vs s001:" not in out


def test_runconfig_diff_filters_max_epochs():
    """trainer.max_epochs bumps are how users add training, not recipe
    changes — filter them from the diff line."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={"model.lr": 3e-4, "trainer.max_epochs": 30},
        prev_runconfig={"model.lr": 3e-4, "trainer.max_epochs": 10},
        prev_session_id="s001",
    )
    assert "vs s001:" not in out


def test_runconfig_diff_shows_real_change_but_hides_noise():
    """A real recipe change still surfaces; only the noise key is filtered
    from the diff continuation line."""
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig={"model.lr": 3e-4, "seed_everything": 12345},
        prev_runconfig={"model.lr": 1e-4, "seed_everything": 67890},
        prev_session_id="s001",
    )
    diff = _diff_line(out)
    assert "vs s001:" in diff
    assert "model.lr" in diff
    assert "seed_everything" not in diff


def test_runconfig_row_caps_diff_at_n_keys():
    """Big diffs truncate with a '+N more' tail."""
    # k0 has the same value on both sides (0 == 0) so doesn't diff;
    # k1..k9 = 9 differing keys; row caps at 4 → '+5 more'.
    curr = {f"model.k{i}": i for i in range(10)}
    prev = {f"model.k{i}": -i for i in range(10)}
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[], ckpt_path=None,
        curr_runconfig=curr,
        prev_runconfig=prev,
        prev_session_id="s001",
    )
    assert "+5 more" in out


# ── backtrack row ─────────────────────────────────────────────────────────

def _bt_render(tmp_path, *, version=None, hset=None, ckpt_path=None,
               latest_version=None, latest_hset=None, latest_step=None):
    """Helper: render the plain briefing in backtrack mode with the given
    latest_* hints. Returns the rendered string."""
    from mlops.hset_state import Mode
    return render_briefing_plain(
        Path("m.py"),
        version if version is not None else _version(),
        hset if hset is not None else _hset(),
        sessions=[_session()],
        ckpt_path=ckpt_path,
        mode=Mode.BACKTRACK,
        latest_version=latest_version,
        latest_hset=latest_hset,
        latest_step=latest_step,
    )


def test_backtrack_row_step_only(tmp_path):
    ckpt = tmp_path / "step=100.ckpt"
    ckpt.write_text("")
    out = _bt_render(
        tmp_path, ckpt_path=ckpt,
        latest_version=_version(),  # same version
        latest_hset=_hset(),         # same hset
        latest_step=999,             # newer step exists
    )
    assert "backtrack" in out
    assert "step=100.ckpt" in out
    assert "step=999" in out
    assert "--new-hset" in out  # consequence-nudge


def test_backtrack_row_older_hset(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_text("")
    out = _bt_render(
        tmp_path,
        version=_version(name="v001"),
        hset=_hset(name="h003"),
        ckpt_path=ckpt,
        latest_version=_version(name="v001"),
        latest_hset=_hset(name="h006"),
        latest_step=None,
    )
    assert "resumed h003" in out
    assert "h006" in out
    assert "untouched" in out


def test_backtrack_row_older_version(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_text("")
    out = _bt_render(
        tmp_path,
        version=_version(name="v002"),
        ckpt_path=ckpt,
        latest_version=_version(name="v005"),
        latest_hset=_hset(),
        latest_step=None,
    )
    assert "resumed v002" in out
    assert "v005" in out
    assert "later versions are untouched" in out


def test_backtrack_row_combined_version_and_hset(tmp_path):
    ckpt = tmp_path / "step=100.ckpt"
    ckpt.write_text("")
    out = _bt_render(
        tmp_path,
        version=_version(name="v002"),
        hset=_hset(name="h003"),
        ckpt_path=ckpt,
        latest_version=_version(name="v005"),
        latest_hset=_hset(name="h006"),
        latest_step=999,
    )
    # Combined headline: 'resumed v002/h003 @ step 100'
    assert "resumed v002/h003" in out
    assert "step 100" in out
    assert "v005/h006" in out
    assert "this branch continues independently" in out


def test_backtrack_row_absent_for_fresh_and_resume_modes(tmp_path):
    """The backtrack row only appears when mode == BACKTRACK."""
    # Fresh
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(), sessions=[], ckpt_path=None,
    )
    assert "backtrack:" not in out
    # Resume
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_text("")
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(),
        sessions=[_session()], ckpt_path=ckpt,
    )
    assert "backtrack:" not in out


def test_samples_preview_caps_to_3_pairs(tmp_path):
    md = "".join(
        f"**Prompt:** `p{i}`\n\np{i} cont{i}\n\n---\n\n"
        for i in range(6)
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "step_000050.md").write_text(md)
    out = render_briefing_plain(
        Path("m.py"), _version(), _hset(dir=tmp_path),
        sessions=[], ckpt_path=None,
    )
    # First three pairs render; the rest are dropped.
    for i in range(3):
        assert f"cont{i}" in out
    for i in range(3, 6):
        assert f"cont{i}" not in out
