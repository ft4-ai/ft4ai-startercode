"""Regression tests for bugs found in May 14 stress test."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from mlops.sampling import write_sample_file


# ── BUG-1: UnicodeEncodeError in write_sample_file ─────────────────────────

class TestWriteSampleFileHandlesSurrogates:
    """LangGen's tokenizer uses errors='surrogateescape' to handle invalid
    UTF-8 byte sequences from an untrained model. The escaped sequences
    appear as surrogate code points (e.g., \\udcaf) in the generated string,
    which the default UTF-8 encoder rejects. write_sample_file must
    handle this gracefully — model output is allowed to be lax; I/O isn't.
    """

    def test_surrogates_do_not_raise(self, tmp_path):
        text = "Once upon a time \udcaf there was a story \udcde end."
        # Pre-fix this raises UnicodeEncodeError. Post-fix it returns cleanly.
        path = write_sample_file(tmp_path, step=0, text=text)
        assert path.exists()

    def test_surrogates_are_replaced(self, tmp_path):
        """Surrogates become '?' on UTF-8 encode with errors='replace'
        (Python's encoder uses '?' for unencodable; U+FFFD is the
        decode-side replacement). Either way, garbage in → markdown-safe
        out; the reader sees '?' as 'unencodable byte was here'."""
        text = "Hello \udcaf world"
        path = write_sample_file(tmp_path, step=0, text=text)
        content = path.read_text(encoding="utf-8")
        assert "\udcaf" not in content
        assert "?" in content
        assert "Hello" in content
        assert "world" in content

    def test_pure_ascii_unchanged(self, tmp_path):
        """The errors='replace' fallback must not alter clean text."""
        text = "## Step 100\n\nA story about a brave little turtle."
        path = write_sample_file(tmp_path, step=100, text=text)
        assert path.read_text(encoding="utf-8") == text

    def test_pure_utf8_unchanged(self, tmp_path):
        """Non-ASCII but valid UTF-8 should pass through unchanged."""
        text = "café — 日本語 — emoji 🎉"
        path = write_sample_file(tmp_path, step=50, text=text)
        assert path.read_text(encoding="utf-8") == text


# ── BUG-3: empty artifacts/ directory in every hset ────────────────────────
# ── BUG-4: "Created new hset tNNN" print is noise ──────────────────────────

class TestCreateNewHsetCleanliness:
    """The hset directory layout per spec rev 7 uses samples/, not
    artifacts/. The stats display covers 'hset: NEW' so the print is
    redundant noise that disrupts the otherwise-cohesive stats block."""

    def test_new_hset_has_no_artifacts_directory(self, tmp_path):
        """Hset dirs created post-rev-7 should never have an artifacts/
        subdir. Pre-fix, _create_new_hset mkdir's it as a leftover from
        the rename from artifacts → samples."""
        from mlops.hset_state import (
            Ft4Args, Signals, VersionInfo, _create_new_hset,
        )
        version = VersionInfo(
            name="v001",
            dir=tmp_path / "v001",
            state_dict_hash="aa",
            model_file_hash="bb",
            description=None,
        )
        version.dir.mkdir()
        signals = Signals("aa", "bb", "cc")
        args = Ft4Args(
            subcommand="train",
            model_file=Path("model.py"),
            model_class=None, hset=None, version=None,
            ckpt="last", new_version=False,
            vdesc=None, hdesc=None, prompt=None,
            lightning_args=[],
        )
        hset = _create_new_hset(version, signals, args)
        assert not (hset.dir / "artifacts").exists()

    def test_new_hset_creates_checkpoints_dir(self, tmp_path):
        """Sanity: we still create checkpoints/ (which Lightning needs)."""
        from mlops.hset_state import (
            Ft4Args, Signals, VersionInfo, _create_new_hset,
        )
        version = VersionInfo(
            name="v001",
            dir=tmp_path / "v001",
            state_dict_hash="aa",
            model_file_hash="bb",
            description=None,
        )
        version.dir.mkdir()
        signals = Signals("aa", "bb", "cc")
        args = Ft4Args(
            subcommand="train", model_file=Path("model.py"),
            model_class=None, hset=None, version=None,
            ckpt="last", new_version=False,
            vdesc=None, hdesc=None, prompt=None,
            lightning_args=[],
        )
        hset = _create_new_hset(version, signals, args)
        assert (hset.dir / "checkpoints").is_dir()

    def test_new_hset_does_not_print(self, tmp_path, capsys):
        """Stats display owns the 'hset: NEW' announcement."""
        from mlops.hset_state import (
            Ft4Args, Signals, VersionInfo, _create_new_hset,
        )
        version = VersionInfo(
            name="v001",
            dir=tmp_path / "v001",
            state_dict_hash="aa",
            model_file_hash="bb",
            description=None,
        )
        version.dir.mkdir()
        signals = Signals("aa", "bb", "cc")
        args = Ft4Args(
            subcommand="train", model_file=Path("model.py"),
            model_class=None, hset=None, version=None,
            ckpt="last", new_version=False,
            vdesc=None, hdesc=None, prompt=None,
            lightning_args=[],
        )
        _create_new_hset(version, signals, args)
        captured = capsys.readouterr()
        assert "Created new hset" not in captured.out
        assert "Created new hset" not in captured.err
