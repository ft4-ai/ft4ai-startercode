# Run the lab:  python -m ft4.demo.measure_entropy
"""
Provides a terminal interface to the measure_entropy engine.

Layout, top to bottom: a stats panel, the story-so-far (rendered whole, with
color), the result of your last guess, and a continuation line that repeats the
recent tail so you're visibly continuing the story. We redraw the whole frame
each round (no live overlay). Input uses stdlib readline for ordinary line
editing; the typed echo is shown bright to match the story.

Everything decided here is presentation; the rules live in engine.py.
"""
import argparse
import random
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ft4.demo.measure_entropy import metrics
from ft4.demo.measure_entropy.engine import (
    MAX_ROUNDS_PER_STORY, NOISY_TRIALS_BELOW, Session, SummaryModel, ViewModel)

TITLE = "Estimating max corpus entropy"

# Token tints in the story-so-far and the verdict. Context/punctuation dim;
# correct guesses green; the true token you missed in red.
_STYLE = {"context": "dim", "correct": "green", "miss": "red", "skip": "dim"}

_ESC = "\033"
_RESET = _ESC + "[0m"


def _entropy_str(m) -> str:
    """The entropy figure for a stats object (ViewModel or SummaryModel, which
    share trials/accuracy/vocab_size): '—' before any trials, 'TODO' if the
    student's metrics.py isn't implemented yet, else 'X.XXX nats'."""
    if m.trials == 0:
        return "—"
    try:
        return f"≤ {metrics.max_corpus_entropy(m.accuracy, m.vocab_size):.3f} nats"
    except Exception:
        return "TODO"


def _metric_cell(vm: ViewModel) -> Text:
    s = _entropy_str(vm)
    return Text(s, style="" if s.endswith("nats") else "dim")


def _render_panel(vm: ViewModel) -> Panel:
    # Each line is a single row with no cross-row alignment, so a plain Text
    # (literal spacing) replaces what a Table.grid would do; the outer grid only
    # stacks the lines vertically.
    head = Text.from_markup(
        f"[bold cyan]trials[/]   [bold]{vm.trials}[/]   "
        f"[bold cyan]accuracy[/]   {vm.correct} / {vm.trials}   ({vm.accuracy * 100:.1f}%)")
    metric = Text.from_markup("[bold cyan]corpus max-entropy[/] ")
    metric.append_text(_metric_cell(vm))
    footer = Text(
        f"V = {vm.vocab_size} (tokenizer vocab)      "
        f"story {vm.story_index}      round {vm.round_num} / {MAX_ROUNDS_PER_STORY}",
        style="dim")

    grid = Table.grid()
    grid.add_row(head)
    grid.add_row("")
    grid.add_row(metric)
    grid.add_row("")
    grid.add_row(footer)
    return Panel(grid, title=TITLE, border_style="cyan", expand=False)


def _render_story(vm: ViewModel) -> Text:
    text = Text()
    for chunk, status in vm.story_segments:
        text.append(chunk, style=_STYLE.get(status, ""))
    return text


def _render_feedback(vm: ViewModel):
    if vm.last_guess is None:
        return None
    g = escape(vm.last_guess)
    if vm.last_correct:
        return Text.from_markup(
            f'\n  You guessed "[bold green]{g}[/bold green]" '
            f'[bold green]✓[/bold green] (correct)')
    t = escape(vm.last_truth or "")
    return Text.from_markup(
        f'\n  You guessed "[bold red]{g}[/bold red]" [bold red]✗[/bold red] '
        f'(incorrect, truth is "[bold]{t}[/bold]")')


def _readline_prompt(tail: str, truncated: bool, continuing: bool) -> str:
    # "❯ " in an accent color, then the repeated tail (bright); we leave the
    # terminal in bright so the typed echo matches. ANSI is wrapped in readline's
    # non-printing markers so cursor math stays correct. Reset after input().
    #   - leading "…" only when there's earlier text we're not showing (#4)
    #   - peek-ahead: a trailing space when a new word starts, none mid-word, so
    #     the cursor sits where the next token actually begins
    accent = "\001" + _ESC + "[36m\002"
    bright = "\001" + _RESET + _ESC + "[1m\002"
    lead = "…" if truncated else ""
    trail = "" if continuing else " "
    return f"{accent}❯ {bright}{lead}{tail}{trail}"


def _banner(console: Console) -> None:
    intro = Text.from_markup(
        "You're standing in for a language model: read the story so far and type how "
        "it continues. \nYour accuracy implies an [bold][italic]upper bound[/italic][/bold] of how much entropy this "
        "corpus carries per token.\n\n"
        "[bold]Each round: type what comes next and press Enter.[/bold] \nUsually that's "
        "the next word; when a word is several tokens you'll finish it one piece at a "
        'time. \nPunctuation is automatic; empty Enter means "the story ends here."'
    )
    console.clear()
    console.print(Panel(intro, title=TITLE, border_style="cyan", expand=False))
    try:
        input("  press Enter to begin … ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit


def _summary(console: Console, m: SummaryModel, seed: int) -> None:
    value = _entropy_str(m)
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="right", style="bold cyan"); body.add_column()
    body.add_row("stories", str(m.stories_seen))
    body.add_row("accuracy", f"{m.correct} / {m.trials}   ({m.accuracy * 100:.1f}%)")
    body.add_row("", "")
    body.add_row("corpus max-entropy", value)
    console.print()
    console.print(Panel(body, title="session over", border_style="cyan", expand=False))
    console.print("Caveats:\n1. This is an [bold][italic]upper bound[/italic][/bold] on the corpus entropy (not an [italic]estimate[/italic])\n2. It assumes you'll achieve the same mean accuracy everywhere")
    if 0 < m.trials < NOISY_TRIALS_BELOW:
        console.print(f"3. Only {m.trials} trials performed; "
                      f"use at least {NOISY_TRIALS_BELOW} trials for better accuracy",
                      highlight=False)
    console.print(f"[dim]Re-run with --seed {seed} to replay these stories.[/dim]",
                  highlight=False)


def _draw_frame(console: Console, vm: ViewModel) -> None:
    console.clear()
    console.print(_render_panel(vm))
    console.print()
    console.print(_render_story(vm))
    feedback = _render_feedback(vm)
    if feedback is not None:
        console.print(feedback)
    console.print()


def main() -> None:
    try:
        import readline  # noqa: F401  (enables line editing for input())
    except ImportError:
        pass

    parser = argparse.ArgumentParser(prog="measure_entropy")
    parser.add_argument("--seed", type=int, default=None,
                        help="story shuffle seed (default: random each run)")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1 << 31)

    console = Console()
    _banner(console)
    session = Session(seed=seed)

    while not session.finished:
        vm = session.view()
        _draw_frame(console, vm)

        if vm.state == "story_done":
            console.print(Text(f"  story complete — {vm.story_correct}/{vm.story_trials} "
                               f"this story", style="cyan"))
            console.print("  [dim]press Enter for the next story →[/dim]", highlight=False)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                break
            session.advance_story()
            continue

        console.print("  type what comes next, then Enter   ·   /q to quit")
        if vm.continuing_word:
            console.print("  [dim]↳ mid-word — type the rest of this word[/dim]",
                          highlight=False)
        elif vm.show_onboarding_cue:
            console.print("  [dim]↳ one piece at a time — usually a whole word[/dim]",
                          highlight=False)
        console.print("\n")
        try:
            entry = input(_readline_prompt(vm.prompt_tail, vm.tail_truncated,
                                           vm.continuing_word))
        except (EOFError, KeyboardInterrupt):
            break
        finally:
            sys.stdout.write(_RESET)        # the prompt left the terminal bright
            sys.stdout.flush()
        if entry.strip() == "/q":
            break
        session.commit(entry)

    _summary(console, session.summary(), seed)


if __name__ == "__main__":
    main()
