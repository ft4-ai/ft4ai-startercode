"""Unit tests for BriefingPrintCallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rich.panel import Panel

from mlops.briefing_print_callback import BriefingPrintCallback


def _patch_console(monkeypatch) -> MagicMock:
    mock_console = MagicMock()
    monkeypatch.setattr(
        "mlops.briefing_print_callback.get_console", lambda: mock_console
    )
    return mock_console


def test_tty_on_train_start_prints_panel(monkeypatch):
    """TTY: print the Rich Panel via get_console."""
    cb = BriefingPrintCallback()
    cb._is_tty = True
    panel = Panel("hello")
    cb.set_briefing(panel, "hello (plain)")
    mock_console = _patch_console(monkeypatch)

    cb.on_train_start(MagicMock(), MagicMock())
    mock_console.print.assert_called_once_with(panel)


def test_non_tty_on_train_start_prints_plain_to_stdout(monkeypatch, capsys):
    """Non-TTY: print the plain string to stdout, NOT the Panel."""
    cb = BriefingPrintCallback()
    cb._is_tty = False
    panel = Panel("hello")
    cb.set_briefing(panel, "ft4 briefing plain text")
    mock_console = _patch_console(monkeypatch)

    cb.on_train_start(MagicMock(), MagicMock())
    mock_console.print.assert_not_called()
    captured = capsys.readouterr()
    assert "ft4 briefing plain text" in captured.out
    assert captured.err == ""


def test_tty_without_panel_skips(monkeypatch):
    cb = BriefingPrintCallback()
    cb._is_tty = True
    mock_console = _patch_console(monkeypatch)
    cb.on_train_start(MagicMock(), MagicMock())
    mock_console.print.assert_not_called()


def test_non_tty_without_plain_skips(monkeypatch, capsys):
    cb = BriefingPrintCallback()
    cb._is_tty = False
    cb.on_train_start(MagicMock(), MagicMock())
    assert capsys.readouterr().out == ""


def test_setup_captures_stdout_tty_for_fit(monkeypatch):
    cb = BriefingPrintCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="fit")
    assert cb._is_tty is True


def test_setup_captures_stdout_non_tty_for_fit(monkeypatch):
    cb = BriefingPrintCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = False
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="fit")
    assert cb._is_tty is False


def test_setup_skips_non_fit_stage(monkeypatch):
    cb = BriefingPrintCallback()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    monkeypatch.setattr("sys.stdout", fake_stdout)
    cb.setup(MagicMock(), MagicMock(), stage="validate")
    assert cb._is_tty is False
