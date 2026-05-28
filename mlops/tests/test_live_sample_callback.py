"""Unit tests for LiveSampleCallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rich.panel import Panel

from mlops.live_sample_callback import (
    LiveSampleCallback,
    _build_body,
    _build_panel,
    _build_title,
    _compute_pair_budget,
    extract_pairs,
    _format_lines_plain,
    _format_sample_plain,
)

# Default test budget — matches the old _MAX_PAIR_CHARS=180 cap so existing
# truncation assertions keep their original arithmetic.
_TEST_PAIR_BUDGET = 180


_SAMPLE_MD = """# Samples at step 42

**Prompt:** `once upon a time`

once upon a time and the cat sat on the mat

---

**Prompt:** `the cat sat on the`

the cat sat on the mat and looked at the moon

---
"""


# ── extract_pairs ────────────────────────────────────────────────────────

def testextract_pairs_basic():
    pairs = extract_pairs(_SAMPLE_MD)
    assert len(pairs) == 2
    assert pairs[0] == ("once upon a time", " and the cat sat on the mat")
    assert pairs[1] == ("the cat sat on the", " mat and looked at the moon")


def testextract_pairs_empty_on_garbage():
    assert extract_pairs("no prompts here") == []


# ── _build_body ───────────────────────────────────────────────────────────

def test_build_body_renders_one_bullet_per_pair():
    body = _build_body(_SAMPLE_MD, _TEST_PAIR_BUDGET)
    text = body.plain
    assert text.count("•") == 2
    assert "•once upon a time" in text
    assert "•the cat sat on the" in text


def test_build_body_truncates_long_pairs_with_ellipsis():
    long_continuation = " " + "x " * 200  # ~400 chars
    md = f"**Prompt:** `tiny`\n\ntiny{long_continuation}\n\n---\n"
    body = _build_body(md, _TEST_PAIR_BUDGET)
    text = body.plain
    assert "…" in text
    assert len(text) <= _TEST_PAIR_BUDGET + 5


# ── _compute_pair_budget ──────────────────────────────────────────────────

def test_compute_pair_budget_wide_terminal_one_line():
    """A 160-col terminal -> single-line budget (~panel-inner - margin)."""
    b = _compute_pair_budget(160)
    # panel_inner = 160 - 4 = 156; >= 120 so single multiplier; -3 margin.
    assert 140 <= b <= 156


def test_compute_pair_budget_narrow_terminal_two_lines():
    """An 80-col terminal -> two lines worth of budget."""
    b = _compute_pair_budget(80)
    # panel_inner = 76; < 120 so 2x; -3 margin.
    assert 140 <= b <= 160  # ~149


def test_compute_pair_budget_floor():
    """Tiny or zero widths floor at _WIDTH_FLOOR."""
    b = _compute_pair_budget(5)
    assert b > 0
    # Floor=60, panel_inner=56, 2x=112, -3=109
    assert b == 109


def test_compute_pair_budget_handles_none():
    b = _compute_pair_budget(None)  # type: ignore[arg-type]
    assert b > 0


# ── _build_title ──────────────────────────────────────────────────────────

def test_build_title_includes_step_and_both_metrics():
    trainer = MagicMock()
    trainer.callback_metrics = {"train_ce_epoch": 1.234, "val_ce_epoch": 0.987}
    title = _build_title(42, trainer)
    text = title.plain
    assert "step 42" in text
    assert "train=1.234" in text
    assert "val=0.987" in text


def test_build_title_prefers_epoch_metric_over_step():
    trainer = MagicMock()
    trainer.callback_metrics = {
        "train_ce_epoch": 1.0,
        "train_ce_step": 5.0,
    }
    title = _build_title(0, trainer)
    assert "train=1.000" in title.plain
    assert "train=5.000" not in title.plain


# ── _format_sample_plain ──────────────────────────────────────────────────

def test_format_sample_plain_header_and_body():
    trainer = MagicMock()
    trainer.callback_metrics = {"train_ce_epoch": 4.541, "val_ce_epoch": 4.560}
    out = _format_sample_plain(628, _SAMPLE_MD, trainer)
    assert out.startswith("=== sample @ step 628")
    assert "train=4.541" in out
    assert "val=4.560" in out
    assert "•once upon a time and the cat sat on the mat" in out
    assert "•the cat sat on the mat and looked at the moon" in out


def test_format_sample_plain_no_metrics():
    trainer = MagicMock()
    trainer.callback_metrics = {}
    out = _format_sample_plain(5, _SAMPLE_MD, trainer)
    assert "=== sample @ step 5 ===" in out
    assert "train=" not in out
    assert "val=" not in out


# ── on_validation_epoch_end ───────────────────────────────────────────────

def _patch_console(monkeypatch) -> MagicMock:
    mock_console = MagicMock()
    # _compute_pair_budget reads get_console().width; give it a real int.
    mock_console.width = 100
    monkeypatch.setattr(
        "mlops.live_sample_callback.get_console", lambda: mock_console
    )
    return mock_console


def _make_trainer(sanity=False, metrics=None):
    trainer = MagicMock()
    trainer.sanity_checking = sanity
    trainer.callback_metrics = metrics or {}
    return trainer


def test_tty_path_prints_panel_via_get_console(monkeypatch):
    """TTY: build a Panel and print via the global console."""
    cb = LiveSampleCallback()
    cb._is_tty = True

    mock_console = _patch_console(monkeypatch)

    pl_module = MagicMock()
    pl_module._ft4_latest_sample = (42, _SAMPLE_MD)

    cb.on_validation_epoch_end(_make_trainer(metrics={"val_ce": 0.5}), pl_module)

    mock_console.print.assert_called_once()
    (arg,), _ = mock_console.print.call_args
    assert isinstance(arg, Panel)


def test_non_tty_path_prints_plain_text_to_stdout(monkeypatch, capsys):
    """Non-TTY (redirected): print plain text to stdout, NOT a Panel."""
    cb = LiveSampleCallback()
    cb._is_tty = False

    mock_console = _patch_console(monkeypatch)

    pl_module = MagicMock()
    pl_module._ft4_latest_sample = (42, _SAMPLE_MD)

    cb.on_validation_epoch_end(_make_trainer(metrics={"train_ce": 4.5, "val_ce": 4.5}), pl_module)

    # No Panel via the rich console.
    mock_console.print.assert_not_called()
    # Plain text on stdout, not stderr.
    captured = capsys.readouterr()
    assert "=== sample @ step 42" in captured.out
    assert "•once upon a time" in captured.out
    assert captured.err == ""


def test_dedups_when_same_step_seen_twice(monkeypatch):
    cb = LiveSampleCallback()
    cb._is_tty = True

    mock_console = _patch_console(monkeypatch)

    pl_module = MagicMock()
    pl_module._ft4_latest_sample = (42, _SAMPLE_MD)
    trainer = _make_trainer()

    cb.on_validation_epoch_end(trainer, pl_module)
    cb.on_validation_epoch_end(trainer, pl_module)
    assert mock_console.print.call_count == 1


def test_sanity_checking_skipped(monkeypatch):
    cb = LiveSampleCallback()
    cb._is_tty = True
    mock_console = _patch_console(monkeypatch)

    pl_module = MagicMock()
    pl_module._ft4_latest_sample = (1, _SAMPLE_MD)

    cb.on_validation_epoch_end(_make_trainer(sanity=True), pl_module)
    mock_console.print.assert_not_called()


def test_no_stashed_sample_skipped(monkeypatch, capsys):
    """If SampleGenerationCallback hasn't stashed anything yet, do nothing."""
    cb = LiveSampleCallback()
    cb._is_tty = False  # would otherwise print plain text

    pl_module = MagicMock(spec=[])  # no _ft4_latest_sample attr

    cb.on_validation_epoch_end(_make_trainer(), pl_module)
    assert capsys.readouterr().out == ""


# ── setup captures isatty ────────────────────────────────────────────────

def test_setup_captures_stdout_tty_for_fit(monkeypatch):
    cb = LiveSampleCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="fit")
    assert cb._is_tty is True


def test_setup_captures_stdout_non_tty_for_fit(monkeypatch):
    cb = LiveSampleCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = False
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="fit")
    assert cb._is_tty is False


def test_setup_skips_non_fit_stage(monkeypatch):
    cb = LiveSampleCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="validate")
    assert cb._is_tty is False


# ── helpers ───────────────────────────────────────────────────────────────

def test_format_lines_plain():
    lines = list(_format_lines_plain(_SAMPLE_MD))
    assert lines == [
        "•once upon a time and the cat sat on the mat",
        "•the cat sat on the mat and looked at the moon",
    ]
