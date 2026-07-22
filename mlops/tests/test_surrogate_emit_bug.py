"""Emit-boundary tests: model output containing invalid UTF-8 bytes.

Regression test for https://ft4ai.atlassian.net/browse/FT-36

Ft4Tokenizer is byte-level BPE, so there is a token for every raw byte, and a sampled
token's bytes need not be valid UTF-8 on their own. LangGen decodes each token separately
with errors='surrogateescape' (a deliberate, byte-faithful round-trip), so such tokens
surface as lone surrogate code points in the yielded str — and lone surrogates crash any
strict-UTF-8 writer (macOS stdout, redirected stdout, files, Rich's console).

Two field reports, one root cause:
  (1) mid-training UnicodeEncodeError in LiveSampleCallback's Rich print;
  (2) `ft4 generate` UnicodeEncodeError at _stream_completions'
      stream.write (char '\\udcda', i.e. raw byte 0xDA).

Test 1 pins the premise and passes before and after the fix. Tests 2-5 specify required
behavior at each emit surface: they fail on the unpatched codebase (reproducing the
field-report crash sites) and pass once emit-boundary sanitization is in place.
"""
from __future__ import annotations

import io

import torch
from rich.console import Console
from types import SimpleNamespace

from ft4.lang_gen import LangGen
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from mlops.generate import _stream_completions
from mlops.live_sample_callback import _build_panel, extract_pairs
from mlops.sampling import generate_sample_markdown

VOCAB_SIZE = Ft4Tokenizer.last_tid + 1

# The script every test uses: normal text, then the lone byte 0xDA from
# field report 2 (a 2-byte lead byte with no continuation — invalid on
# its own), then '\u00a9' ('©', bytes C2 A9) split across two byte-tokens.
SCRIPT_BYTES = [b" on", b"\xda", b"\xc2", b"\xa9", b" mat"]


class ScriptedModel:
    """Returns logits peaked on a scripted token sequence, then RES_STOP.

    LangGen calls the model with the full sequence so far (1, L) and reads
    the last position; L grows by one per generated token, so the script
    index is L minus the length of the initial [BOS] + prompt tokens,
    captured on the first call. Peak-vs--inf logits make each draw
    deterministic (singleton support), as in tests/test_lang_gen.py.
    """

    device = torch.device("cpu")
    training = False  # sampling helpers toggle eval()/train() around us

    def __init__(self, script_bytes: list[bytes]):
        # Each entry tokenizes to one or more real token ids (a lone
        # invalid byte hits its base byte-token; text hits learned merges).
        self._script: list[int] = []
        for b in script_bytes:
            self._script.extend(Ft4Tokenizer.tokenize(b))
        self._base_len: int | None = None

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        if self._base_len is None:
            self._base_len = L
        idx = L - self._base_len
        peak = (self._script[idx] if idx < len(self._script)
                else Ft4Tokenizer.RES_STOP)
        logits = torch.full((B, L, VOCAB_SIZE), float("-inf"))
        logits[..., peak] = 100.0
        return logits


def _strict_stream() -> io.TextIOWrapper:
    """Same strictness as macOS stdout / a redirected log file — the
    regime in which both field reports crashed."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    stream.isatty = lambda: False  # plain-text path; no ANSI prefixes
    return stream


# ----------------------------------------------------------------------
# 1. The premise, pinned at its source (passes before AND after the fix).
# ----------------------------------------------------------------------

def test_langgen_yields_surrogates_for_invalid_byte_tokens():
    """Characterization, not a bug: LangGen's per-token surrogateescape
    decoding is byte-faithful by design, so a lone 0xDA token yields the
    surrogate U+DCDA. This is the input condition every emit surface
    must survive; if this test ever fails, LangGen's contract changed
    and the emit-boundary tests below are testing the wrong hazard."""
    lg = LangGen(model=ScriptedModel(SCRIPT_BYTES), initial_txt="The cat sat",
                 max_to_generate=40)
    out = "".join(lg)
    assert "\udcda" in out                    # the field-report byte
    assert "\udcc2" in out and "\udca9" in out  # the split '©'


# ----------------------------------------------------------------------
# 2-5. Required behavior at each emit surface (fail on unpatched code).
# ----------------------------------------------------------------------

def test_generate_stream_survives_invalid_byte_token():
    """Field report 2: `ft4 generate` must stream to a strict stdout
    without crashing (generate.py: stream.write(token_str))."""
    stream = _strict_stream()

    _stream_completions(ScriptedModel(SCRIPT_BYTES), ["The cat sat"], stream,
                        max_to_generate=40, temperature=None, min_p=None)

    stream.flush()
    out = stream.buffer.getvalue().decode("utf-8")  # strict round-trip
    assert "The cat sat" in out


def test_sample_markdown_is_strict_utf8_encodable():
    """Root cause of field report 1: the markdown stashed for the live
    panel and written to sample files must be strict-UTF-8 encodable."""
    md = generate_sample_markdown(ScriptedModel(SCRIPT_BYTES),
                                  prompts=["The cat sat"], step=3)

    md.encode("utf-8")  # must not raise — this is the crash, reduced to one line


def test_live_panel_prints_to_strict_console():
    """Field report 1's exact crash site: mid-training, LiveSampleCallback
    prints the sample panel via Rich to a strict-UTF-8 console."""
    md = generate_sample_markdown(ScriptedModel(SCRIPT_BYTES),
                                  prompts=["The cat sat"], step=3)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    console = Console(file=stream, force_terminal=True, width=80)
    trainer = SimpleNamespace(callback_metrics={})

    console.print(_build_panel(3, md, trainer, per_pair_budget=200))  # must not raise
    stream.flush()


def test_split_char_reassembles_and_lone_byte_replaced():
    """Quality of the fix, not just crash-absence: a valid char split
    across adjacent tokens must reassemble ('©'); a genuinely invalid
    lone byte becomes U+FFFD; no surrogates reach the parser."""
    md = generate_sample_markdown(ScriptedModel(SCRIPT_BYTES),
                                  prompts=["The cat sat"], step=3)

    pairs = extract_pairs(md)
    assert pairs, "panel parser found no prompt/completion pairs"
    completion = pairs[0][1]
    assert "\u00a9" in completion                 # C2 + A9 reassembled into ©
    assert "\ufffd" in completion                 # lone DA honestly replaced
    assert not any("\udc80" <= ch <= "\udcff" for ch in completion)
