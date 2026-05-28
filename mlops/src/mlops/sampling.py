"""ft4's sample-generation glue.

`LangGen` (in `ft4.lang_gen`) is a general-purpose autoregressive
sampler — tokenizer-aware, runnable from a shell. This module is its
ft4-CLI wrapper: it knows about the hset directory layout, the default
prompts ft4 uses, and the markdown format written to
`<hset>/samples/step_NNNNNN.md`.

Models don't need a `generate_samples` method. We just try LangGen; for
non-language models (iris, classifiers, anything LangGen can't drive), the
attempt raises and the caller (callback or run.py) skips. Keeping that
detection a broad try/except is intentional — see the design discussion
in this branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_SAMPLE_PROMPTS = [
    "The cat sat",
    "Billy was cold,",
    "The evil wizard",
]


def generate_completions(
    model,
    prompts: list[str],
    *,
    max_to_generate: int = 128,
    temperature: float | None = None,
    min_p: float | None = None,
) -> list[tuple[str, str]]:
    """For each prompt, return `(prompt, completion)`. Pure helper used by
    both `generate_sample_markdown` (during training) and `ft4 generate`
    (the standalone command).

    `temperature` / `min_p` of None means "use LangGen's default" — the CLI
    layer chooses whether to expose them; LangGen owns the numerical
    defaults.

    Toggles model.eval() for the duration, restoring `model.train()` if the
    caller had it in train mode. Raises whatever LangGen raises (typically
    AttributeError on non-language models); the caller decides how to
    handle that.
    """
    from ft4.lang_gen import LangGen

    lg_kwargs: dict = {"max_to_generate": max_to_generate}
    if temperature is not None:
        lg_kwargs["temperature"] = temperature
    if min_p is not None:
        lg_kwargs["min_p"] = min_p

    was_training = model.training
    model.eval()
    try:
        out: list[tuple[str, str]] = []
        for prompt in prompts:
            response = LangGen(model=model, initial_txt=prompt, **lg_kwargs)
            completion = "".join(response)
            out.append((prompt, completion))
        return out
    finally:
        if was_training:
            model.train()


def generate_sample_markdown(
    model,
    prompts: Optional[list[str]] = None,
    step: int = 0,
    max_to_generate: int = 40,
) -> str:
    """Generate samples from `model` for each prompt; return markdown.

    Thin wrapper around `generate_completions` that wraps each pair in the
    `**Prompt:** ... \\n\\n<prompt><completion>\\n\\n---` shape that
    LiveSampleCallback and SampleGenerationCallback both consume.
    """
    if prompts is None:
        prompts = DEFAULT_SAMPLE_PROMPTS
    pairs = generate_completions(model, prompts, max_to_generate=max_to_generate)
    out_lines = [f"# Samples at step {step}", ""]
    for prompt, completion in pairs:
        out_lines.append(f"**Prompt:** `{prompt}`")
        out_lines.append("")
        out_lines.append(f"{prompt}{completion}")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")
    return "\n".join(out_lines)


def write_sample_file(hset_dir: Path, step: int, text: str) -> Path:
    """Write a sample markdown file at <hset>/samples/step_NNNNNN.md.

    Creates <hset>/samples/ on first call. Overwrites existing files at
    the same step (safe: same step means same model state modulo
    sample-time nondeterminism, which is bounded).

    `text` is allowed to contain unencodable code points (e.g., surrogates
    from a tokenizer using errors='surrogateescape' on a model whose
    early-training output isn't yet valid UTF-8). They're replaced with
    U+FFFD in the output.

    Returns the file path.
    """
    samples_dir = hset_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    path = samples_dir / f"step_{step:06d}.md"
    path.write_text(text, encoding="utf-8", errors="replace")
    return path
