"""Unit tests for hset_state.create_session_dir.

(Could be merged into test_hset_state.py; kept separate here for clarity
of the step-7 v2 delivery — feel free to inline.)
"""
import pytest

from mlops.hset_state import create_session_dir


def test_first_session_creates_s001(tmp_path):
    sd = create_session_dir(tmp_path)
    assert sd == tmp_path / "sessions" / "s001"
    assert sd.is_dir()


def test_creates_sessions_parent_if_missing(tmp_path):
    create_session_dir(tmp_path)
    assert (tmp_path / "sessions").is_dir()


def test_second_call_increments(tmp_path):
    s1 = create_session_dir(tmp_path)
    s2 = create_session_dir(tmp_path)
    assert s1.name == "s001"
    assert s2.name == "s002"


def test_continues_from_highest_existing(tmp_path):
    """If the user manually wiped s002 (or any non-max number), the next
    call continues from max+1, not the first gap."""
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "s001").mkdir()
    (tmp_path / "sessions" / "s003").mkdir()
    sd = create_session_dir(tmp_path)
    assert sd.name == "s004"


def test_ignores_non_session_dirs(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "scratch").mkdir()
    (tmp_path / "sessions" / "s001").mkdir()
    sd = create_session_dir(tmp_path)
    assert sd.name == "s002"


def test_returned_dir_is_empty(tmp_path):
    sd = create_session_dir(tmp_path)
    assert list(sd.iterdir()) == []


def test_handles_three_digit_overflow(tmp_path):
    """s999 → s1000 should still produce a directory; the regex permits
    4+ digits."""
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "s999").mkdir()
    sd = create_session_dir(tmp_path)
    assert sd.name == "s1000"
