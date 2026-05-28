"""LiveSampleCallback.

Renders the latest generated samples after each validation. Two paths:

  * TTY (stdout is a terminal): build a Rich Panel and print it via the
    global console. Lightning's Live region is active during fit(); Rich's
    render-hook contract routes the print to scrollback above the
    progress bar.
  * Non-TTY (stdout redirected to a file or pipe): print a plain-text
    block — a header line plus one bulleted line per (prompt, completion)
    pair. No box characters, no width assumptions.

Source for both: `pl_module._ft4_latest_sample`, which
`SampleGenerationCallback` stashes after each write.
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional

import lightning as L
from rich import get_console
from rich.panel import Panel
from rich.text import Text


_PROMPT_STYLE = "italic dim"
_BULLET = "•"
# Floor and ceiling for the panel width we trust. Floor keeps tiny terminals
# from producing absurd budgets; ceiling avoids generating multi-thousand-char
# lines on very wide displays.
_WIDTH_FLOOR = 60
_WIDTH_CEILING = 240
# At this panel-inner width we switch from two lines of sample per pair
# (narrow displays) to one (wide displays). Matches ISSUES.md guidance.
_WIDE_THRESHOLD = 120


class LiveSampleCallback(L.Callback):
    """Per-validation: Rich panel on TTY, plain-text block when redirected."""

    def __init__(self) -> None:
        super().__init__()
        self._last_step_shown: Optional[int] = None
        # Captured at setup, before Lightning's Live region replaces sys.stdout
        # with a rich.file_proxy.FileProxy. FileProxy inherits io.TextIOBase,
        # whose isatty() returns False and shadows FileProxy's __getattr__ — so
        # sys.stdout.isatty() reads as non-TTY for the duration of fit().
        self._is_tty = False

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "fit":
            self._is_tty = sys.stdout.isatty()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        latest = getattr(pl_module, "_ft4_latest_sample", None)
        if latest is None:
            return
        step, text = latest
        if step == self._last_step_shown:
            return
        self._last_step_shown = step

        if self._is_tty:
            # Live is active — Rich's render hook routes this above the
            # progress bar in the scrollback.
            per_pair_budget = _compute_pair_budget(get_console().width)
            get_console().print(_build_panel(step, text, trainer, per_pair_budget))
        else:
            # Redirected: emit plain text to stdout (where the user is
            # capturing). No panel borders, no fixed-width assumptions.
            print(_format_sample_plain(step, text, trainer))


# ── Rich panel rendering ──────────────────────────────────────────────────

def _build_panel(step: int, sample_markdown: str, trainer, per_pair_budget: int) -> Panel:
    title = _build_title(step, trainer)
    body = _build_body(sample_markdown, per_pair_budget)
    return Panel(body, title=title, title_align="left", border_style="magenta")


def _compute_pair_budget(console_width: int | None) -> int:
    """Per-(prompt, completion) character budget for the Rich panel.

    Wide displays (>= _WIDE_THRESHOLD) get one line per pair; narrow get two.
    The 3-char trailing margin reserves space for the truncation `…`.
    """
    width = console_width if isinstance(console_width, int) and console_width > 0 else _WIDTH_FLOOR
    width = max(_WIDTH_FLOOR, min(width, _WIDTH_CEILING))
    panel_inner = width - 4  # 2 borders + ~2 padding
    if panel_inner < 1:
        panel_inner = 1
    multiplier = 1 if panel_inner >= _WIDE_THRESHOLD else 2
    return max(1, multiplier * panel_inner - 3)


def _build_title(step: int, trainer) -> Text:
    t = Text()
    t.append("sample ", style="bold")
    t.append(f"@ step {step}", style="dim")
    train_v, val_v = _train_val_metrics(trainer)
    if train_v is not None or val_v is not None:
        t.append("  ", style="dim")
        first = True
        if train_v is not None:
            t.append(f"train={train_v:.3f}", style="dim")
            first = False
        if val_v is not None:
            if not first:
                t.append("  ", style="dim")
            t.append(f"val={val_v:.3f}", style="dim")
    return t


def _train_val_metrics(trainer) -> tuple[Optional[float], Optional[float]]:
    metrics = getattr(trainer, "callback_metrics", {}) or {}
    return (
        _metric(metrics, ("train_ce", "train_loss", "loss")),
        _metric(metrics, ("val_ce", "val_loss")),
    )


def _metric(metrics: dict, bases: tuple[str, ...]) -> Optional[float]:
    """Pick the best matching metric for the title display.

    Mirrors run.py's _pick_metric priority (epoch > "" > step), without
    importing it (this module shouldn't depend on the smart-flow surface).
    """
    for base in bases:
        for suffix in ("_epoch", "", "_step"):
            key = f"{base}{suffix}"
            v = metrics.get(key)
            if v is None:
                continue
            try:
                return float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                continue
    return None


def _build_body(sample_markdown: str, per_pair_budget: int) -> Text:
    """One bullet-led line per prompt/completion pair, `italic dim` prompt
    plus default-styled completion. Each pair is capped at `per_pair_budget`
    chars so the panel height stays predictable."""
    pairs = extract_pairs(sample_markdown)
    if not pairs:
        return Text(sample_markdown[:max(0, per_pair_budget)])

    out = Text()
    for i, (prompt, continuation) in enumerate(pairs):
        if i:
            out.append("\n")
        out.append(_BULLET, style="dim")
        out.append(prompt, style=_PROMPT_STYLE)
        budget = per_pair_budget - len(prompt) - len(_BULLET)
        if budget < 0:
            budget = 0
        if len(continuation) > budget:
            out.append(continuation[:max(0, budget - 1)])
            out.append("…", style="dim")
        else:
            out.append(continuation)
    return out


def extract_pairs(markdown: str) -> list[tuple[str, str]]:
    """Parse `(prompt, continuation)` pairs from the sample markdown.

    The markdown follows this shape per prompt:

        **Prompt:** `<prompt>`

        <prompt><completion>

        ---
    """
    pairs: list[tuple[str, str]] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("**Prompt:**"):
            backtick = line.find("`")
            end = line.rfind("`")
            if backtick == -1 or end <= backtick:
                i += 1
                continue
            prompt = line[backtick + 1 : end]
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                full = lines[j]
                continuation = full[len(prompt):] if full.startswith(prompt) else full
                pairs.append((prompt, continuation))
                i = j + 1
                continue
        i += 1
    return pairs


# ── plain-text rendering ──────────────────────────────────────────────────

def _format_sample_plain(step: int, sample_markdown: str, trainer) -> str:
    """Plain-text block matching the Rich panel's content."""
    bits = [f"step {step}"]
    train_v, val_v = _train_val_metrics(trainer)
    if train_v is not None:
        bits.append(f"train={train_v:.3f}")
    if val_v is not None:
        bits.append(f"val={val_v:.3f}")
    header = f"=== sample @ {'  '.join(bits)} ==="
    body = "\n".join(_format_lines_plain(sample_markdown))
    return f"{header}\n{body}"


def _format_lines_plain(sample_markdown: str) -> Iterable[str]:
    """One bullet-led line per (prompt, completion) pair. No truncation —
    log files have room."""
    pairs = extract_pairs(sample_markdown)
    if not pairs:
        yield sample_markdown
        return
    for prompt, continuation in pairs:
        yield f"{_BULLET}{prompt}{continuation}"
