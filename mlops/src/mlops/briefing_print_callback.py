"""BriefingPrintCallback.

Prints the ft4 briefing once at on_train_start. Two paths:

  * TTY (stdout is a terminal): print the Rich Panel via the global
    console. Lightning's Live region is active by then; Rich's render-hook
    contract routes the print to scrollback above the progress bar.
  * Non-TTY (stdout redirected): print the plain-text version directly to
    stdout — no panel borders, no width assumptions.

Wiring: run.py builds both forms in Phase 0 and passes them via
`set_briefing(panel, plain)` before trainer.fit().
"""
from __future__ import annotations

import sys
from typing import Optional

import lightning as L
from rich import get_console
from rich.panel import Panel


class BriefingPrintCallback(L.Callback):
    """Print the briefing once at train_start; Rich panel on TTY, plain text otherwise."""

    def __init__(self) -> None:
        super().__init__()
        self._panel: Optional[Panel] = None
        self._plain: Optional[str] = None
        # Captured at setup, before Lightning's Live region replaces sys.stdout
        # with a rich.file_proxy.FileProxy. FileProxy inherits io.TextIOBase,
        # whose isatty() returns False and shadows FileProxy's __getattr__ — so
        # sys.stdout.isatty() reads as non-TTY for the duration of fit().
        self._is_tty = False

    def set_briefing(self, panel: Panel, plain: str) -> None:
        """Called by run.py with both rendered forms of the briefing."""
        self._panel = panel
        self._plain = plain

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "fit":
            self._is_tty = sys.stdout.isatty()

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._is_tty:
            if self._panel is None:
                return
            # Live is active — Rich's render hook routes this above the
            # progress bar in the scrollback.
            get_console().print(self._panel)
        else:
            if self._plain is None:
                return
            print(self._plain)
