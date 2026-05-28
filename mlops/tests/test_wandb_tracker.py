"""Unit tests for wandb_tracker.

Covers `make_wandb_logger`'s three failure modes — wandb missing, not
logged in, and any other init error — each surfaces an Ft4UserError
with a helpful, --wandb-mentioning message. No network calls; all wandb
interactions go through fake modules installed in sys.modules.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mlops.hset_state import Ft4UserError, HsetInfo, VersionInfo


@pytest.fixture
def version_hset(tmp_path: Path) -> tuple[VersionInfo, HsetInfo, Path]:
    """Minimal VersionInfo / HsetInfo / session_dir for make_wandb_logger."""
    version_dir = tmp_path / "v001"
    version_dir.mkdir()
    hset_dir = version_dir / "h001"
    hset_dir.mkdir()
    session_dir = hset_dir / "sessions" / "s001"
    session_dir.mkdir(parents=True)
    return (
        VersionInfo(name="v001", dir=version_dir,
                    state_dict_hash="aa", model_file_hash="bb"),
        HsetInfo(name="h001", dir=hset_dir, config_hash="cc"),
        session_dir,
    )


def _install_fake_wandb(monkeypatch, *, api_key: str | None = "secret") -> ModuleType:
    """Install a fake `wandb` module so `import wandb` inside
    make_wandb_logger picks it up without touching the network."""
    fake = ModuleType("wandb")
    fake.api = SimpleNamespace(api_key=api_key)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_not_installed_raises_helpful_error(monkeypatch, version_hset):
    """ImportError on `import wandb` becomes Ft4UserError with install hint."""
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes import raise
    version, hset, session_dir = version_hset

    from mlops.wandb_tracker import make_wandb_logger
    with pytest.raises(Ft4UserError) as exc_info:
        make_wandb_logger("transformer", version, hset, session_dir)
    msg = str(exc_info.value)
    assert "not installed" in msg
    assert "--wandb" in msg  # always tell them they can drop the flag


def test_not_logged_in_raises_helpful_error(monkeypatch, version_hset):
    _install_fake_wandb(monkeypatch, api_key=None)
    version, hset, session_dir = version_hset

    from mlops.wandb_tracker import make_wandb_logger
    with pytest.raises(Ft4UserError) as exc_info:
        make_wandb_logger("transformer", version, hset, session_dir)
    msg = str(exc_info.value)
    assert "not logged in" in msg
    assert "wandb login" in msg
    assert "--wandb" in msg


def test_logger_init_failure_surfaces_error(monkeypatch, version_hset):
    """If WandbLogger.experiment raises (e.g., auth fails on wandb.init()),
    surface the exception class and message; still suggest dropping --wandb."""
    _install_fake_wandb(monkeypatch, api_key="secret")
    version, hset, session_dir = version_hset

    class _Boom:
        def __init__(self, **kw):
            self._kw = kw
        @property
        def experiment(self):
            raise RuntimeError("network unreachable")

    import lightning.pytorch.loggers as lp_loggers
    monkeypatch.setattr(lp_loggers, "WandbLogger", _Boom)

    from mlops.wandb_tracker import make_wandb_logger
    with pytest.raises(Ft4UserError) as exc_info:
        make_wandb_logger("transformer", version, hset, session_dir)
    msg = str(exc_info.value)
    assert "RuntimeError" in msg
    assert "network unreachable" in msg
    assert "--wandb" in msg
