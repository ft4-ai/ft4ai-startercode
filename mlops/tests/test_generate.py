"""Tests for `ft4 generate`.

Mix of pure-function unit tests (warning logic, output rendering, default
prompts) and integration tests (no-checkpoint error path via a real
`prepare_run` on a fresh runs_root). The actual `LangGen` call is mocked
in the integration tests since the course tokenizer fixture isn't always
present and isn't the thing under test.
"""
from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mlops.hset_state import Ft4Args, Ft4UserError
from mlops.generate import (
    _DEFAULT_MAX_TOKENS,
    _load_checkpoint,
    _maybe_warn_undertrained,
    _model_default_prompts,
    _stream_completions,
    run_generate,
)
from mlops.run import run
from mlops.sampling import DEFAULT_SAMPLE_PROMPTS


# ── _maybe_warn_undertrained ──────────────────────────────────────────────

def test_warn_when_no_sessions():
    buf = io.StringIO()
    _maybe_warn_undertrained([], stream=buf)
    msg = buf.getvalue()
    assert "no completed training sessions" in msg
    assert "--force" in msg


def test_warn_when_zero_epoch_interrupted():
    buf = io.StringIO()
    _maybe_warn_undertrained(
        [{"status": "interrupted", "epochs": 0, "steps": 5}],
        stream=buf,
    )
    assert "interrupted" in buf.getvalue()


def test_warn_when_zero_epoch_completed():
    buf = io.StringIO()
    _maybe_warn_undertrained(
        [{"status": "completed", "epochs": 0, "steps": 5}],
        stream=buf,
    )
    assert "fewer than one epoch" in buf.getvalue()


def test_no_warn_when_trained():
    buf = io.StringIO()
    _maybe_warn_undertrained(
        [{"status": "completed", "epochs": 3, "steps": 200}],
        stream=buf,
    )
    assert buf.getvalue() == ""


def test_no_warn_when_multiple_sessions_with_progress():
    """A history of small-epoch sessions still warns only on cumulative
    epochs == 0. With any total > 0, stay silent."""
    buf = io.StringIO()
    _maybe_warn_undertrained(
        [
            {"status": "interrupted", "epochs": 0, "steps": 5},
            {"status": "completed", "epochs": 2, "steps": 100},
        ],
        stream=buf,
    )
    assert buf.getvalue() == ""


# ── _stream_completions ──────────────────────────────────────────────────

class _FakeStream:
    """Captures writes; reports isatty per the constructor flag."""
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty
        self.buf = io.StringIO()

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, s: str) -> int:
        return self.buf.write(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self.buf.getvalue()


class _FakeLangGen:
    """Iterable yielding a fixed sequence of token strings — stand-in for
    LangGen in the streaming-output tests."""
    def __init__(self, model=None, initial_txt: str = "", **_kw):
        # Tokens encode the prompt so multi-prompt tests can tell them apart.
        self._tokens = list(f"-stub-{initial_txt}")

    def __iter__(self):
        return iter(self._tokens)


class _FakeModel:
    """Minimal model API needed by _stream_completions (training flag + eval/train)."""
    def __init__(self):
        self.training = True
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.train_called = True
        self.training = mode
        return self


def test_stream_single_prompt_no_bullet_plain(monkeypatch):
    monkeypatch.setattr("ft4.lang_gen.LangGen", _FakeLangGen)
    s = _FakeStream(is_tty=False)
    _stream_completions(
        _FakeModel(), ["Once "], s,
        max_to_generate=99, temperature=None, min_p=None,
    )
    assert s.getvalue() == "Once -stub-Once \n"


def test_stream_multi_prompt_uses_bullets_plain(monkeypatch):
    monkeypatch.setattr("ft4.lang_gen.LangGen", _FakeLangGen)
    s = _FakeStream(is_tty=False)
    _stream_completions(
        _FakeModel(), ["A ", "B "], s,
        max_to_generate=99, temperature=None, min_p=None,
    )
    out = s.getvalue().splitlines()
    assert out == ["• A -stub-A ", "• B -stub-B "]


def test_stream_tty_path_emits_ansi(monkeypatch):
    """TTY branch wraps the prompt in ANSI italic-dim escapes."""
    monkeypatch.setattr("ft4.lang_gen.LangGen", _FakeLangGen)
    s = _FakeStream(is_tty=True)
    _stream_completions(
        _FakeModel(), ["Hi "], s,
        max_to_generate=99, temperature=None, min_p=None,
    )
    out = s.getvalue()
    assert "\x1b[" in out  # some ANSI escape present
    assert "Hi" in out
    assert "-stub-Hi" in out


def test_stream_restores_train_mode(monkeypatch):
    """Streaming switches model to eval() but restores train() when it was set."""
    monkeypatch.setattr("ft4.lang_gen.LangGen", _FakeLangGen)
    m = _FakeModel()
    _stream_completions(
        m, ["x"], _FakeStream(is_tty=False),
        max_to_generate=99, temperature=None, min_p=None,
    )
    assert m.eval_called is True
    assert m.train_called is True
    assert m.training is True


# ── _model_default_prompts ────────────────────────────────────────────────

class _ModelWithDefaults(nn.Module):
    DEFAULT_SAMPLE_PROMPTS = ["alpha", "beta"]


class _ModelWithoutDefaults(nn.Module):
    pass


def test_default_prompts_uses_class_attr():
    m = _ModelWithDefaults()
    assert _model_default_prompts(m) == ["alpha", "beta"]


def test_default_prompts_falls_back_to_ft4_defaults():
    m = _ModelWithoutDefaults()
    assert _model_default_prompts(m) == list(DEFAULT_SAMPLE_PROMPTS)


# ── _load_checkpoint ──────────────────────────────────────────────────────

def test_load_checkpoint_loads_state_dict(tmp_path):
    """Lightning checkpoints store weights under 'state_dict'. Verify the
    helper loads them and toggles eval mode."""
    src = nn.Linear(4, 2)
    # Mutate so the snapshot differs from a fresh init.
    with torch.no_grad():
        src.weight.fill_(1.5)
        src.bias.fill_(-0.5)
    ckpt = tmp_path / "fake.ckpt"
    torch.save({"state_dict": src.state_dict()}, ckpt)

    dst = nn.Linear(4, 2)
    dst.train()  # ensure _load_checkpoint flips to eval
    _load_checkpoint(dst, ckpt)
    assert dst.training is False
    assert torch.allclose(dst.weight, src.weight)
    assert torch.allclose(dst.bias, src.bias)


def test_load_checkpoint_rejects_non_lightning_ckpt(tmp_path):
    """A checkpoint dict without 'state_dict' should error clearly."""
    ckpt = tmp_path / "bad.ckpt"
    torch.save({"other": 1}, ckpt)
    with pytest.raises(Ft4UserError, match="state_dict"):
        _load_checkpoint(nn.Linear(4, 2), ckpt)


# ── Integration: no-checkpoint error path ─────────────────────────────────

_TINY_MODEL_SRC = textwrap.dedent("""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import lightning as L

    class TinyModel(L.LightningModule):
        DEFAULT_SAMPLE_PROMPTS = ["unit", "test"]
        def __init__(self, hidden: int = 8):
            super().__init__()
            self.save_hyperparameters()
            self.fc = nn.Linear(4, hidden)
            self.out = nn.Linear(hidden, 2)
        def forward(self, x):
            return self.out(self.fc(x))
        def training_step(self, batch, _):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss)
            return loss
        def validation_step(self, batch, _):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("val_loss", loss)
            return loss
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)


    class TinyDataModule(L.LightningDataModule):
        def setup(self, stage=None):
            torch.manual_seed(0)
            x = torch.randn(16, 4)
            y = torch.randint(0, 2, (16,))
            self.train_ds = TensorDataset(x[:12], y[:12])
            self.val_ds = TensorDataset(x[12:], y[12:])
        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=4)
        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=4)
""")


_MODEL_YAML = textwrap.dedent("""
    data:
      class_path: tiny.TinyDataModule
""")


_DEFAULTS_YAML = textwrap.dedent("""
    trainer:
      max_epochs: 1
      enable_progress_bar: false
      enable_model_summary: false
      logger:
        class_path: lightning.pytorch.loggers.CSVLogger
        init_args:
          save_dir: "."
          name: ""
          version: ""
      callbacks:
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            dirpath: "."
            filename: "{epoch}"
            save_last: true
            save_top_k: -1
            monitor: null
        - class_path: mlops.session_finalization.SessionFinalizationCallback
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "tiny.py"
    model_file.write_text(_TINY_MODEL_SRC)
    (tmp_path / "tiny.yaml").write_text(_MODEL_YAML)
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(_DEFAULTS_YAML)
    import mlops.main as main_mod
    monkeypatch.setattr(main_mod, "_default_cascade_paths", lambda _mf: [defaults, _mf.with_suffix(".yaml")])
    sys.modules.pop("tiny", None)
    yield model_file, tmp_path / "runs"
    sys.modules.pop("tiny", None)


def test_generate_errors_on_fresh_hset(env, monkeypatch):
    """No prior run → resolve_version fails (read op, no version exists)."""
    model_file, runs_root = env
    args = Ft4Args(subcommand="generate", model_file=model_file)
    with pytest.raises(Ft4UserError):
        run_generate(args, runs_root=runs_root)


def test_generate_runs_end_to_end_with_mocked_langgen(env, monkeypatch, capsys):
    """Train briefly so a checkpoint exists, then verify run_generate calls
    the streaming primitive with the right kwargs and writes output to
    stdout. The streaming function itself is mocked — its behavior isn't
    what's under test here."""
    model_file, runs_root = env
    # Train once to populate v001/h001/checkpoints/last.ckpt.
    run(Ft4Args(subcommand="train", model_file=model_file), runs_root=runs_root)
    capsys.readouterr()  # discard training output

    captured: dict = {}

    def fake_stream(model, prompts, stream, **kw):
        captured["prompts"] = list(prompts)
        captured["kwargs"] = dict(kw)
        for p in prompts:
            stream.write(f"• {p}-stub-{p}\n")

    monkeypatch.setattr("mlops.generate._stream_completions", fake_stream)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    args = Ft4Args(
        subcommand="generate",
        model_file=model_file,
        prompt=["alpha", "beta"],
        max_tokens=42,
        temperature=0.5,
        min_p=0.3,
        force=True,
    )
    code = run_generate(args, runs_root=runs_root)
    assert code == 0
    assert captured["prompts"] == ["alpha", "beta"]
    assert captured["kwargs"]["max_to_generate"] == 42
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["min_p"] == 0.3

    out = capsys.readouterr().out.splitlines()
    assert out == ["• alpha-stub-alpha", "• beta-stub-beta"]


def test_generate_uses_model_default_prompts_when_none_given(env, monkeypatch, capsys):
    """No --prompt → falls back to the model's DEFAULT_SAMPLE_PROMPTS."""
    model_file, runs_root = env
    run(Ft4Args(subcommand="train", model_file=model_file), runs_root=runs_root)
    capsys.readouterr()

    captured: dict = {}

    def fake_stream(model, prompts, stream, **kw):
        captured["prompts"] = list(prompts)

    monkeypatch.setattr("mlops.generate._stream_completions", fake_stream)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    args = Ft4Args(subcommand="generate", model_file=model_file, force=True)
    run_generate(args, runs_root=runs_root)
    # TinyModel.DEFAULT_SAMPLE_PROMPTS = ["unit", "test"]
    assert captured["prompts"] == ["unit", "test"]


def test_generate_uses_default_max_tokens_when_unspecified(env, monkeypatch, capsys):
    """Default --max-tokens is _DEFAULT_MAX_TOKENS (2048)."""
    model_file, runs_root = env
    run(Ft4Args(subcommand="train", model_file=model_file), runs_root=runs_root)
    capsys.readouterr()

    captured: dict = {}
    def fake_stream(model, prompts, stream, **kw):
        captured.update(kw)
    monkeypatch.setattr("mlops.generate._stream_completions", fake_stream)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    args = Ft4Args(
        subcommand="generate", model_file=model_file,
        prompt=["x"], force=True,
    )
    run_generate(args, runs_root=runs_root)
    assert captured["max_to_generate"] == _DEFAULT_MAX_TOKENS
