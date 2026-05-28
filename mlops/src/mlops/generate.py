"""ft4 generate — produce completions from a trained checkpoint.

Strictly read-only: never appends to `sessions.jsonl`, never writes under
`samples/`, never bumps `last_session_at`. Resolves a checkpoint via the
same Phase 0 that `ft4 train` and `ft4 show` use, loads the weights, and
prints prompt+completion pairs to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import torch

from mlops.hset_state import Ft4Args, Ft4UserError
from mlops.prepare import PreparedRun, prepare_run
from mlops.sampling import DEFAULT_SAMPLE_PROMPTS


_DEFAULT_MAX_TOKENS = 2048


def run_generate(args: Ft4Args, *, runs_root: Path = Path("runs")) -> int:
    p = prepare_run(args, runs_root=runs_root)

    if p.ckpt_path is None:
        raise Ft4UserError(
            f"No checkpoint found for hset {p.hset.name}. "
            f"Run `ft4 train {args.model_file}` first."
        )

    _load_checkpoint(p.model, p.ckpt_path)

    if not args.force:
        _maybe_warn_undertrained(p.sessions, stream=sys.stderr)

    prompts = args.prompt or _model_default_prompts(p.model)
    if not prompts:
        raise Ft4UserError(
            "No prompt given and the model defines no defaults. "
            "Use --prompt 'your prompt' (repeatable)."
        )

    _stream_completions(
        p.model,
        prompts,
        sys.stdout,
        max_to_generate=args.max_tokens or _DEFAULT_MAX_TOKENS,
        temperature=args.temperature,
        min_p=args.min_p,
    )
    return 0


# ── Checkpoint loading ───────────────────────────────────────────────────

def _load_checkpoint(model, ckpt_path: Path) -> None:
    """Load the `state_dict` from a Lightning checkpoint into `model`.

    Lightning checkpoints are a dict containing `state_dict` plus optimizer,
    schedulers, callbacks, etc. — we only want the weights. `map_location`
    defers to the model's existing device; the trainer that `prepare_run`
    built has already placed weights on the right device.
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise Ft4UserError(
            f"checkpoint at {ckpt_path} has no 'state_dict' key — "
            f"is this a Lightning checkpoint?"
        )
    model.load_state_dict(state_dict)
    model.eval()


def _model_default_prompts(model) -> list[str]:
    """Per-model override (a `DEFAULT_SAMPLE_PROMPTS` attribute on the class)
    falls back to ft4's built-in defaults."""
    cls_default = getattr(type(model), "DEFAULT_SAMPLE_PROMPTS", None)
    if cls_default:
        return list(cls_default)
    return list(DEFAULT_SAMPLE_PROMPTS)


# ── Undertrained-checkpoint warning ──────────────────────────────────────

def _maybe_warn_undertrained(sessions: list[dict], *, stream: TextIO) -> None:
    """Heuristic warnings for checkpoints that probably aren't worth
    generating from. To stderr so piped output stays clean."""
    if not sessions:
        print(
            "Warning: no completed training sessions recorded — output may "
            "be gibberish. Use --force to silence.",
            file=stream,
        )
        return
    total_epochs = sum(int(s.get("epochs") or 0) for s in sessions)
    last = sessions[-1]
    if total_epochs == 0:
        if last.get("status") in {"interrupted", "error"}:
            print(
                "Warning: training was interrupted before completing an "
                "epoch. Use --force to silence.",
                file=stream,
            )
        else:
            print(
                "Warning: fewer than one epoch of training so far. "
                "Use --force to silence.",
                file=stream,
            )


# ── Streaming output ─────────────────────────────────────────────────────

_BULLET = "• "
# ANSI escapes for italic-dim prompt on a TTY (matches the styling that the
# live sample panel uses during training). Direct escapes keep the streaming
# path free of Rich's buffering machinery.
_ANSI_ITALIC_DIM = "\x1b[3;2m"
_ANSI_DIM = "\x1b[2m"
_ANSI_RESET = "\x1b[0m"


def _stream_completions(
    model,
    prompts: list[str],
    stream: TextIO,
    *,
    max_to_generate: int,
    temperature: float | None,
    min_p: float | None,
) -> None:
    """Stream each LangGen token to `stream` as it's produced.

    Single prompt -> prompt then completion, no bullet. Multiple prompts ->
    each on its own line, prefixed with `• `. TTY gets italic-dim prompt
    via ANSI escapes; non-TTY emits raw text.

    Each token is flushed immediately so the user sees output appear in
    real time rather than after the full completion. Mid-stream exceptions
    are visible up to the point of failure, then propagate.
    """
    # Lazy import so non-generate paths don't pay the LangGen import cost.
    from ft4.lang_gen import LangGen

    is_tty = hasattr(stream, "isatty") and stream.isatty()
    use_bullet = len(prompts) > 1

    lg_kwargs: dict = {"max_to_generate": max_to_generate}
    if temperature is not None:
        lg_kwargs["temperature"] = temperature
    if min_p is not None:
        lg_kwargs["min_p"] = min_p

    was_training = model.training
    model.eval()
    try:
        for prompt in prompts:
            _write_prompt_prefix(prompt, stream, is_tty=is_tty, use_bullet=use_bullet)
            try:
                lg = LangGen(model=model, initial_txt=prompt, **lg_kwargs)
                for token_str in lg:
                    stream.write(token_str)
                    stream.flush()
            finally:
                # Newline ends the prompt's output even on mid-stream exception,
                # so the next prompt (or trailing error) starts cleanly.
                stream.write("\n")
                stream.flush()
    finally:
        if was_training:
            model.train()


def _write_prompt_prefix(prompt: str, stream: TextIO, *, is_tty: bool, use_bullet: bool) -> None:
    if is_tty:
        if use_bullet:
            stream.write(f"{_ANSI_DIM}{_BULLET}{_ANSI_RESET}")
        stream.write(f"{_ANSI_ITALIC_DIM}{prompt}{_ANSI_RESET}")
    else:
        if use_bullet:
            stream.write(_BULLET)
        stream.write(prompt)
    stream.flush()
