"""Unit tests for hset_state.

Tests for `inject_paths` and `persist_config` (the Lightning seam) are
deferred to integration testing in step 5; the underlying mutation
contract is already pinned by mlops/tests/contract/test_modelcheckpoint_dirpath_mutation.py
and test_csvlogger_save_dir_mutation.py.
"""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch.nn as nn

from mlops.hset_state import (
    Ft4Args,
    Ft4UserError,
    Signals,
    RUNCONFIG_STD_KEYS,
    VersionInfo,
    _flatten,
    all_hsets,
    all_versions,
    append_session,
    capture_git,
    compute_signals,
    diff_hset_configs,
    hash_config,
    hash_file,
    hash_state_dict,
    read_sessions,
    resolve_checkpoint,
    resolve_hset,
    resolve_version,
    update_hset_last_session,
)


# ── Test models ────────────────────────────────────────────────────────────

class _SmallModel(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 2)


class _DifferentArchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_args(model_file: Path, **kw) -> Ft4Args:
    return Ft4Args(
        subcommand=kw.pop("subcommand", "train"),
        model_file=model_file,
        **kw,
    )


def _write_model_file(path: Path, content: str = "# placeholder\n") -> Path:
    path.write_text(content)
    return path


def _signals_for(model: nn.Module, model_file: Path, config: dict) -> Signals:
    return compute_signals(model, model_file, config)


# ── hash_state_dict ────────────────────────────────────────────────────────

def test_hash_state_dict_deterministic():
    # Same architecture, different random init → same hash.
    h1 = hash_state_dict(_SmallModel(8))
    h2 = hash_state_dict(_SmallModel(8))
    assert h1 == h2


def test_hash_state_dict_sensitive_to_size():
    assert hash_state_dict(_SmallModel(8)) != hash_state_dict(_SmallModel(16))


def test_hash_state_dict_sensitive_to_architecture():
    assert hash_state_dict(_SmallModel(4)) != hash_state_dict(_DifferentArchModel())


# ── hash_file ──────────────────────────────────────────────────────────────

def test_hash_file_deterministic(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("hello")
    assert hash_file(p) == hash_file(p)


def test_hash_file_sensitive_to_content(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("foo")
    b = tmp_path / "b.py"
    b.write_text("bar")
    assert hash_file(a) != hash_file(b)


# ── _flatten ───────────────────────────────────────────────────────────────

def test_flatten_basic():
    assert _flatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_nested():
    assert _flatten({"trainer": {"max_epochs": 4, "lr": 3e-4}}) == {
        "trainer.max_epochs": 4,
        "trainer.lr": 3e-4,
    }


def test_flatten_list_left_intact():
    out = _flatten({"trainer": {"callbacks": [{"a": 1}, {"b": 2}]}})
    assert out == {"trainer.callbacks": [{"a": 1}, {"b": 2}]}


# ── hash_config ────────────────────────────────────────────────────────────

def test_hash_config_deterministic():
    cfg = {"model": {"dim": 512, "lr": 3e-4}}
    assert hash_config(cfg, frozenset()) == hash_config(cfg, frozenset())


def test_hash_config_excludes_stopping_keys():
    cfg1 = {"trainer": {"lr": 3e-4, "max_epochs": 4}, "model": {"dim": 512}}
    cfg2 = {"trainer": {"lr": 3e-4, "max_epochs": 10}, "model": {"dim": 512}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_sensitive_to_nonstopping_keys():
    cfg1 = {"model": {"dim": 512}}
    cfg2 = {"model": {"dim": 1024}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) != hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_callback_list_changes():
    """Adding/removing/reordering callbacks does not fork a hset."""
    cfg1 = {"model": {"dim": 512}, "trainer": {"callbacks": [
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint"},
    ]}}
    cfg2 = {"model": {"dim": 512}, "trainer": {"callbacks": [
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint"},
        {"class_path": "mlops.session_finalization.SessionFinalizationCallback"},
    ]}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_logger_changes():
    """Switching logger (CSVLogger save_dir, etc.) doesn't fork a hset."""
    cfg1 = {"model": {"dim": 512}, "trainer": {"logger": {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": "runs", "name": ""}
    }}}
    cfg2 = {"model": {"dim": 512}, "trainer": {"logger": {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": "elsewhere", "name": "x"}
    }}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_ui_flags():
    cfg1 = {"model": {"dim": 512}, "trainer": {"enable_progress_bar": True}}
    cfg2 = {"model": {"dim": 512}, "trainer": {"enable_progress_bar": False}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_still_sensitive_to_scientific_trainer_keys():
    """gradient_clip_val is scientifically meaningful — keep it in the hash."""
    cfg1 = {"model": {"dim": 512}, "trainer": {"gradient_clip_val": 1.0}}
    cfg2 = {"model": {"dim": 512}, "trainer": {"gradient_clip_val": 2.0}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) != hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_precision():
    """trainer.precision moved to the exclusion list: a precision flip is
    treated as infrastructure rather than a forked experiment."""
    cfg3 = {"model": {"dim": 512}, "trainer": {"precision": "bf16-mixed"}}
    cfg4 = {"model": {"dim": 512}, "trainer": {"precision": "32-true"}}
    assert hash_config(cfg3, RUNCONFIG_STD_KEYS) == hash_config(cfg4, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_seed_everything():
    """Changing the random seed shouldn't fork the hset."""
    cfg1 = {"model": {"dim": 512}, "seed_everything": 1}
    cfg2 = {"model": {"dim": 512}, "seed_everything": 999_999}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_batch_size_and_dataloader_infra():
    """Mini-batch size and DataLoader perf flags don't fork the hset."""
    common_model = {"model": {"dim": 512}}
    cfg1 = {**common_model, "data": {"init_args": {"batch_size": 32, "num_workers": 0}}}
    cfg2 = {**common_model, "data": {"init_args": {"batch_size": 256, "num_workers": 8, "pin_memory": True}}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_config_paths():
    """The --config layering (stored under `config`) varies by cwd; not science."""
    cfg1 = {"model": {"dim": 512}, "config": ["/a/defaults.yaml"]}
    cfg2 = {"model": {"dim": 512}, "config": ["/b/defaults.yaml", "/c/model.yaml"]}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_insensitive_to_dict_ordering():
    cfg1 = {"model": {"a": 1, "b": 2}}
    cfg2 = {"model": {"b": 2, "a": 1}}
    assert hash_config(cfg1, frozenset()) == hash_config(cfg2, frozenset())


def test_hash_config_sensitive_to_list_ordering():
    cfg1 = {"callbacks": [{"name": "a"}, {"name": "b"}]}
    cfg2 = {"callbacks": [{"name": "b"}, {"name": "a"}]}
    assert hash_config(cfg1, frozenset()) != hash_config(cfg2, frozenset())


def test_hash_config_ignores_lr_change():
    """LR is a runconfig knob, not part of the hset identity."""
    cfg1 = {"model": {"dim": 512, "lr": 3e-4}}
    cfg2 = {"model": {"dim": 512, "lr": 1e-4}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_weight_decay_change():
    cfg1 = {"model": {"dim": 512, "weight_decay": 1e-2}}
    cfg2 = {"model": {"dim": 512, "weight_decay": 0.0}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_ignores_warmup_steps_change():
    cfg1 = {"model": {"dim": 512, "warmup_steps": 1000}}
    cfg2 = {"model": {"dim": 512, "warmup_steps": 3000}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_lr_under_init_args_also_ignored():
    """Same rule for both subclass-mode (model.init_args.lr) and bare-mode."""
    cfg1 = {"model": {"init_args": {"dim": 512, "lr": 3e-4}}}
    cfg2 = {"model": {"init_args": {"dim": 512, "lr": 1e-4}}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


def test_hash_config_still_sensitive_to_dim_and_seq_len():
    """Architecture shape (dim) and data shape (seq_len) stay in the hset
    hash — these change the experiment's identity."""
    cfg1 = {"model": {"dim": 512}, "data": {"init_args": {"seq_len": 128}}}
    cfg2 = {"model": {"dim": 1024}, "data": {"init_args": {"seq_len": 128}}}
    cfg3 = {"model": {"dim": 512}, "data": {"init_args": {"seq_len": 256}}}
    base = hash_config(cfg1, RUNCONFIG_STD_KEYS)
    assert hash_config(cfg2, RUNCONFIG_STD_KEYS) != base
    assert hash_config(cfg3, RUNCONFIG_STD_KEYS) != base


def test_hash_config_ignores_dropout_change():
    """Dropout is a regularization recipe knob, not part of the hset identity:
    same shapes, same ckpt, just a different training trajectory."""
    cfg1 = {"model": {"dim": 512, "dropout": 0.1}}
    cfg2 = {"model": {"dim": 512, "dropout": 0.2}}
    assert hash_config(cfg1, RUNCONFIG_STD_KEYS) == hash_config(cfg2, RUNCONFIG_STD_KEYS)


# ── resolve_version, write op ──────────────────────────────────────────────

def test_resolve_version_first_call_creates_v001(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    args = _make_args(model_file, subcommand="train")
    sig = _signals_for(_SmallModel(), model_file, {})
    v = resolve_version(sig, args, runs_root=tmp_path / "runs")
    assert v.name == "v001"
    assert (v.dir / "version.json").exists()
    assert (v.dir / "model.py.snapshot").exists()
    assert (v.dir / "model.py.snapshot").read_bytes() == model_file.read_bytes()


def test_resolve_version_same_signals_reuses(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    args = _make_args(model_file, subcommand="train")
    sig = _signals_for(_SmallModel(), model_file, {})
    v1 = resolve_version(sig, args, runs_root=tmp_path / "runs")
    v2 = resolve_version(sig, args, runs_root=tmp_path / "runs")
    assert v1.name == v2.name == "v001"


def test_resolve_version_changed_file_forks(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py", "# v1\n")
    args = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(), model_file, {})
    v1 = resolve_version(sig1, args, runs_root=tmp_path / "runs")

    model_file.write_text("# v2\n")
    sig2 = _signals_for(_SmallModel(), model_file, {})
    v2 = resolve_version(sig2, args, runs_root=tmp_path / "runs")
    assert v1.name == "v001"
    assert v2.name == "v002"


def test_resolve_version_changed_arch_stays_in_same_version(tmp_path):
    """A version is the model's *code* identity. Changing parameter shapes
    (e.g., --model.dim) under the same source file is a new HSET inside
    the existing version, not a new version. Only a model.py edit forks
    a new version."""
    model_file = _write_model_file(tmp_path / "my_model.py")
    args = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(8), model_file, {})
    v1 = resolve_version(sig1, args, runs_root=tmp_path / "runs")
    # Different param shapes under the SAME model file.
    sig2 = _signals_for(_SmallModel(16), model_file, {})
    v2 = resolve_version(sig2, args, runs_root=tmp_path / "runs")
    assert v1.name == "v001"
    assert v2.name == "v001", (
        f"expected reuse of v001 (model file unchanged); got {v2.name}"
    )


def test_resolve_version_file_edit_forks_new_version(tmp_path):
    """The flip side: when the source file changes, that IS a new version,
    even if the architecture is identical."""
    model_file = _write_model_file(tmp_path / "my_model.py", "# v1\n")
    args = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(), model_file, {})
    v1 = resolve_version(sig1, args, runs_root=tmp_path / "runs")
    assert v1.name == "v001"

    # File content changes; architecture unchanged.
    model_file.write_text("# v2 — different comment\n")
    sig2 = _signals_for(_SmallModel(), model_file, {})
    assert sig2.model_file_hash != sig1.model_file_hash
    v2 = resolve_version(sig2, args, runs_root=tmp_path / "runs")
    assert v2.name == "v002"


def test_resolve_version_write_back_to_original_arch_reuses_v001(tmp_path):
    """User's reproduced bug in miniature: train at arch A, train at arch
    B (forks h002 within v001 — different state_dict), then train at arch
    A again. The third run must land back in v001, not create v003."""
    model_file = _write_model_file(tmp_path / "my_model.py")
    args = _make_args(model_file, subcommand="train")
    sig_small = _signals_for(_SmallModel(8), model_file, {})
    sig_big = _signals_for(_SmallModel(16), model_file, {})

    v1 = resolve_version(sig_small, args, runs_root=tmp_path / "runs")
    v2 = resolve_version(sig_big, args, runs_root=tmp_path / "runs")
    v3 = resolve_version(sig_small, args, runs_root=tmp_path / "runs")
    assert v1.name == "v001"
    assert v2.name == "v001"
    assert v3.name == "v001", (
        f"expected reuse of v001 on arch-A return; got {v3.name}"
    )


def test_resolve_version_new_version_flag_always_forks(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args = _make_args(model_file, subcommand="train")
    v1 = resolve_version(sig, args, runs_root=tmp_path / "runs")
    args_forced = _make_args(model_file, subcommand="train", new_version=True)
    v2 = resolve_version(sig, args_forced, runs_root=tmp_path / "runs")
    assert v1.name == "v001"
    assert v2.name == "v002"


def test_resolve_version_vdesc_set_on_create(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args = _make_args(model_file, subcommand="train", vdesc="initial attempt")
    v = resolve_version(sig, args, runs_root=tmp_path / "runs")
    assert v.description == "initial attempt"
    data = json.loads((v.dir / "version.json").read_text())
    assert data["description"] == "initial attempt"


def test_resolve_version_vdesc_updates_existing(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args1 = _make_args(model_file, subcommand="train")
    v1 = resolve_version(sig, args1, runs_root=tmp_path / "runs")
    assert v1.description is None
    args2 = _make_args(model_file, subcommand="train", vdesc="updated")
    v2 = resolve_version(sig, args2, runs_root=tmp_path / "runs")
    assert v2.name == v1.name
    assert v2.description == "updated"


# ── resolve_version, read op ──────────────────────────────────────────────

def test_resolve_version_read_finds_compatible(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args_w = _make_args(model_file, subcommand="train")
    v_created = resolve_version(sig, args_w, runs_root=tmp_path / "runs")

    args_r = _make_args(model_file, subcommand="generate")
    v_found = resolve_version(sig, args_r, runs_root=tmp_path / "runs")
    assert v_found.name == v_created.name


def test_resolve_version_read_errors_when_none_compatible(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    args = _make_args(model_file, subcommand="generate")
    sig = _signals_for(_SmallModel(), model_file, {})
    with pytest.raises(Ft4UserError, match="no compatible model version"):
        resolve_version(sig, args, runs_root=tmp_path / "runs")


def test_resolve_version_read_errors_when_file_changed(tmp_path):
    """Version identity is now strictly the model file. Read mode used to
    warn-and-load when state_dict matched but the file had drifted; that
    fallback is gone in favor of clean semantics. Users with drifted
    code can pin --version vNNN explicitly to force a load."""
    model_file = _write_model_file(tmp_path / "my_model.py", "# v1\n")
    args_w = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(), model_file, {})
    resolve_version(sig1, args_w, runs_root=tmp_path / "runs")

    # Same architecture, file content changed.
    model_file.write_text("# v1 — comment edit\n")
    sig2 = _signals_for(_SmallModel(), model_file, {})
    assert sig2.state_dict_hash == sig1.state_dict_hash
    assert sig2.model_file_hash != sig1.model_file_hash

    args_r = _make_args(model_file, subcommand="generate")
    with pytest.raises(Ft4UserError, match="no compatible model version"):
        resolve_version(sig2, args_r, runs_root=tmp_path / "runs")


def test_resolve_version_pin_allows_arch_change(tmp_path):
    """--version vNNN pins the model FILE identity. Different param
    shapes under the same file are fine (they become different hsets
    inside vNNN); only model-file drift errors out."""
    model_file = _write_model_file(tmp_path / "my_model.py")
    args_w = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(8), model_file, {})
    resolve_version(sig1, args_w, runs_root=tmp_path / "runs")

    sig2 = _signals_for(_SmallModel(16), model_file, {})
    args_pin = _make_args(model_file, subcommand="generate", version="v001")
    v = resolve_version(sig2, args_pin, runs_root=tmp_path / "runs")
    assert v.name == "v001"


def test_resolve_version_pin_errors_on_file_change(tmp_path):
    """A --version vNNN pin must still reject a drifted model file —
    that's a different version by definition."""
    model_file = _write_model_file(tmp_path / "my_model.py", "# v1\n")
    args_w = _make_args(model_file, subcommand="train")
    sig1 = _signals_for(_SmallModel(), model_file, {})
    resolve_version(sig1, args_w, runs_root=tmp_path / "runs")

    model_file.write_text("# v1 — comment edit\n")
    sig2 = _signals_for(_SmallModel(), model_file, {})
    assert sig2.model_file_hash != sig1.model_file_hash

    args_pin = _make_args(model_file, subcommand="generate", version="v001")
    with pytest.raises(Ft4UserError, match="model file has changed"):
        resolve_version(sig2, args_pin, runs_root=tmp_path / "runs")


def test_resolve_version_pin_errors_when_missing(tmp_path):
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args_pin = _make_args(model_file, subcommand="generate", version="v999")
    (tmp_path / "runs" / "my_model").mkdir(parents=True)
    with pytest.raises(Ft4UserError, match="does not exist"):
        resolve_version(sig, args_pin, runs_root=tmp_path / "runs")


# ── resolve_hset ──────────────────────────────────────────────────────────

def _setup_version(tmp_path) -> tuple[Path, VersionInfo, Signals]:
    """Helper: create v001 and return (model_file, version, signals)."""
    model_file = _write_model_file(tmp_path / "my_model.py")
    sig = _signals_for(_SmallModel(), model_file, {})
    args = _make_args(model_file, subcommand="train")
    v = resolve_version(sig, args, runs_root=tmp_path / "runs")
    return model_file, v, sig


def test_resolve_hset_first_call_creates_t001(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train")
    t = resolve_hset(v, sig, args)
    assert t.name == "h001"
    assert (t.dir / "hset.json").exists()
    assert (t.dir / "checkpoints").exists()
    # samples/ is created lazily by write_sample_file, not by hset creation
    assert not (t.dir / "artifacts").exists()  # removed in spec rev 7
    data = json.loads((t.dir / "hset.json").read_text())
    assert data["config_hash"] == sig.config_hash


def test_resolve_hset_same_config_hash_reuses(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train")
    t1 = resolve_hset(v, sig, args)
    t2 = resolve_hset(v, sig, args)
    assert t1.name == t2.name == "h001"


def test_resolve_hset_different_config_forks(tmp_path):
    model_file, v, _ = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train")
    sig1 = Signals(state_dict_hash="x", model_file_hash="y", config_hash="aaa")
    sig2 = Signals(state_dict_hash="x", model_file_hash="y", config_hash="bbb")
    t1 = resolve_hset(v, sig1, args)
    t2 = resolve_hset(v, sig2, args)
    assert t1.name == "h001"
    assert t2.name == "h002"


def test_resolve_hset_pin_errors_on_config_mismatch_for_write(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train")
    resolve_hset(v, sig, args)

    sig2 = Signals(
        state_dict_hash=sig.state_dict_hash,
        model_file_hash=sig.model_file_hash,
        config_hash="different",
    )
    args_pin = _make_args(model_file, subcommand="train", hset="h001")
    with pytest.raises(Ft4UserError, match="does not reproduce"):
        resolve_hset(v, sig2, args_pin)


def test_resolve_hset_pin_works_on_config_mismatch_for_read(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train")
    resolve_hset(v, sig, args)

    sig2 = Signals(
        state_dict_hash=sig.state_dict_hash,
        model_file_hash=sig.model_file_hash,
        config_hash="different",
    )
    args_pin = _make_args(model_file, subcommand="generate", hset="h001")
    t = resolve_hset(v, sig2, args_pin)
    assert t.name == "h001"


def test_resolve_hset_pin_errors_when_hset_missing(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args_pin = _make_args(model_file, subcommand="generate", hset="t999")
    with pytest.raises(Ft4UserError, match="does not exist"):
        resolve_hset(v, sig, args_pin)


def test_resolve_hset_hdesc_set_on_create(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args = _make_args(model_file, subcommand="train", hdesc="trying small dim")
    t = resolve_hset(v, sig, args)
    assert t.description == "trying small dim"


def test_resolve_hset_hdesc_updates_existing(tmp_path):
    model_file, v, sig = _setup_version(tmp_path)
    args1 = _make_args(model_file, subcommand="train")
    t1 = resolve_hset(v, sig, args1)
    args2 = _make_args(model_file, subcommand="train", hdesc="updated")
    t2 = resolve_hset(v, sig, args2)
    assert t1.name == t2.name
    assert t2.description == "updated"


def test_resolve_hset_new_hset_forces_fresh_even_when_hash_matches(tmp_path):
    """--new-hset bypasses the config-hash lookup so the user can branch off
    an old ckpt without overwriting the existing line of training."""
    model_file, v, sig = _setup_version(tmp_path)
    args1 = _make_args(model_file, subcommand="train")
    t1 = resolve_hset(v, sig, args1)
    args2 = _make_args(model_file, subcommand="train", new_hset=True)
    t2 = resolve_hset(v, sig, args2)
    assert t1.name == "h001"
    assert t2.name == "h002"
    assert t1.config_hash == t2.config_hash


# ── resolve_checkpoint ─────────────────────────────────────────────────────

def test_resolve_checkpoint_last_exists(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "last.ckpt").write_bytes(b"dummy")
    assert resolve_checkpoint(tmp_path, "last") == ckpt_dir / "last.ckpt"


def test_resolve_checkpoint_last_missing(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    assert resolve_checkpoint(tmp_path, "last") is None


def test_resolve_checkpoint_best_errors(tmp_path):
    with pytest.raises(Ft4UserError, match="best.*deferred"):
        resolve_checkpoint(tmp_path, "best")


def test_resolve_checkpoint_explicit_path(tmp_path):
    p = tmp_path / "mychpt.ckpt"
    p.write_bytes(b"dummy")
    assert resolve_checkpoint(tmp_path, str(p)) == p


def test_resolve_checkpoint_explicit_path_missing(tmp_path):
    with pytest.raises(Ft4UserError, match="not found"):
        resolve_checkpoint(tmp_path, str(tmp_path / "nope.ckpt"))


# ── sessions.jsonl ─────────────────────────────────────────────────────────

def test_append_and_read_sessions(tmp_path):
    rec1 = {"session": 1, "started_at": "2026-05-11T10:00:00+00:00"}
    rec2 = {"session": 2, "started_at": "2026-05-11T11:00:00+00:00"}
    append_session(tmp_path, rec1)
    append_session(tmp_path, rec2)
    assert read_sessions(tmp_path) == [rec1, rec2]


def test_read_sessions_when_missing(tmp_path):
    assert read_sessions(tmp_path) == []


def test_update_hset_last_session(tmp_path):
    (tmp_path / "hset.json").write_text(json.dumps({
        "created_at": "2026-05-11T10:00:00+00:00",
        "last_session_at": "2026-05-11T10:00:00+00:00",
        "description": None,
        "config_hash": "abc",
    }))
    new_time = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    update_hset_last_session(tmp_path, new_time)
    data = json.loads((tmp_path / "hset.json").read_text())
    assert data["last_session_at"] == new_time.isoformat()


# ── capture_git ────────────────────────────────────────────────────────────

GIT_AVAILABLE = shutil.which("git") is not None


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def _commit(path: Path, file_name: str = "f.txt", content: str = "x") -> None:
    (path / file_name).write_text(content)
    subprocess.run(["git", "add", file_name], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path, check=True, capture_output=True,
    )


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_capture_git_outside_repo(tmp_path):
    commit, dirty = capture_git(tmp_path)
    assert commit is None
    assert dirty is None


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_capture_git_clean_repo(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path)
    commit, dirty = capture_git(tmp_path)
    assert commit is not None
    assert len(commit) == 40
    assert dirty is False


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_capture_git_dirty_repo(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    _, dirty = capture_git(tmp_path)
    assert dirty is True


# ── compute_signals integration ────────────────────────────────────────────

def test_compute_signals_complete(tmp_path):
    model_file = _write_model_file(tmp_path / "m.py", "# m\n")
    sig = compute_signals(_SmallModel(), model_file, {"model": {"dim": 8}})
    assert sig.state_dict_hash and sig.model_file_hash and sig.config_hash
    # All three are SHA-256 hex digests, 64 chars long.
    assert len(sig.state_dict_hash) == 64
    assert len(sig.model_file_hash) == 64
    assert len(sig.config_hash) == 64


# ── all_versions / all_hsets (scanner helpers) ─────────────────────────────

def _make_version(model_dir: Path, name: str) -> Path:
    v = model_dir / name
    v.mkdir(parents=True)
    (v / "version.json").write_text(json.dumps({
        "state_dict_hash": "sd-" + name,
        "model_file_hash": "mf-" + name,
    }))
    return v


def _make_hset(version_dir: Path, name: str) -> Path:
    h = version_dir / name
    h.mkdir(parents=True)
    (h / "hset.json").write_text(json.dumps({"config_hash": "ch-" + name}))
    return h


def test_all_versions_newest_first(tmp_path):
    model_dir = tmp_path / "model"
    v1 = _make_version(model_dir, "v001")
    v2 = _make_version(model_dir, "v002")
    v3 = _make_version(model_dir, "v003")
    # Force a known mtime order: v002 newest, then v003, then v001.
    import os
    now = datetime.now(timezone.utc).timestamp()
    os.utime(v1, (now - 300, now - 300))
    os.utime(v3, (now - 100, now - 100))
    os.utime(v2, (now, now))
    names = [v.name for v in all_versions(model_dir, newest_first=True)]
    assert names == ["v002", "v003", "v001"]


def test_all_versions_missing_model_dir(tmp_path):
    assert all_versions(tmp_path / "nope") == []


def test_all_versions_skips_dirs_without_version_json(tmp_path):
    model_dir = tmp_path / "m"
    _make_version(model_dir, "v001")
    (model_dir / "v002").mkdir()  # no version.json
    names = [v.name for v in all_versions(model_dir)]
    assert names == ["v001"]


def test_all_hsets_newest_first(tmp_path):
    v = _make_version(tmp_path / "m", "v001")
    h1 = _make_hset(v, "h001")
    h2 = _make_hset(v, "h002")
    import os
    now = datetime.now(timezone.utc).timestamp()
    os.utime(h1, (now - 100, now - 100))
    os.utime(h2, (now, now))
    names = [h.name for h in all_hsets(v, newest_first=True)]
    assert names == ["h002", "h001"]


def test_all_hsets_missing_version_dir(tmp_path):
    assert all_hsets(tmp_path / "v_nope") == []


# ── diff_hset_configs ──────────────────────────────────────────────────────

def _write_yaml(hset_dir: Path, config: dict) -> None:
    import yaml
    hset_dir.mkdir(parents=True, exist_ok=True)
    (hset_dir / "config.yaml").write_text(yaml.safe_dump(config))


def test_diff_hset_configs_no_diff(tmp_path):
    a = tmp_path / "h001"
    b = tmp_path / "h002"
    _write_yaml(a, {"model": {"dim": 8}})
    _write_yaml(b, {"model": {"dim": 8}})
    assert diff_hset_configs(b, a) == ""


def test_diff_hset_configs_one_key(tmp_path):
    base = tmp_path / "h001"
    this = tmp_path / "h002"
    # Use a shape key (dim) since dropout is now a runconfig knob and would
    # be filtered out of the hset diff.
    _write_yaml(base, {"model": {"dim": 8, "depth": 2}})
    _write_yaml(this, {"model": {"dim": 16, "depth": 2}})
    assert diff_hset_configs(this, base) == "model.dim=16"


def test_diff_hset_configs_truncates_to_max_keys(tmp_path):
    base = tmp_path / "h001"
    this = tmp_path / "h002"
    _write_yaml(base, {"a": 1, "b": 2, "c": 3, "d": 4})
    _write_yaml(this, {"a": 10, "b": 20, "c": 30, "d": 40})
    out = diff_hset_configs(this, base, max_keys=2)
    # Alphabetical: a, b kept; c, d in "+2 more".
    assert out == "a=10, b=20, +2 more"


def test_diff_hset_configs_excludes_infra_prefixes(tmp_path):
    base = tmp_path / "h001"
    this = tmp_path / "h002"
    _write_yaml(base, {"trainer": {"max_epochs": 4, "callbacks": []}})
    _write_yaml(this, {"trainer": {"max_epochs": 100, "callbacks": [{"x": 1}]}})
    # Both `max_epochs` and `callbacks` are excluded.
    assert diff_hset_configs(this, base) == ""


def test_diff_hset_configs_missing_files(tmp_path):
    a = tmp_path / "h001"
    b = tmp_path / "h002"
    a.mkdir()
    b.mkdir()  # neither has config.yaml
    assert diff_hset_configs(b, a) == ""


def test_diff_hset_configs_same_dir(tmp_path):
    _write_yaml(tmp_path / "h001", {"a": 1})
    assert diff_hset_configs(tmp_path / "h001", tmp_path / "h001") == ""


# ── extract_runconfig ──────────────────────────────────────────────────────

def test_extract_runconfig_picks_recipe_knobs():
    from mlops.hset_state import extract_runconfig
    cfg = {"model": {"dim": 512, "lr": 3e-4, "weight_decay": 1e-2, "dropout": 0.1}}
    out = extract_runconfig(cfg)
    assert out == {
        "model.dropout": 0.1,
        "model.lr": 3e-4,
        "model.weight_decay": 1e-2,
    }


def test_extract_runconfig_omits_architecture_keys():
    from mlops.hset_state import extract_runconfig
    # dim, depth are shape knobs (hset identity); not in runconfig.
    cfg = {"model": {"dim": 512, "depth": 8}}
    assert extract_runconfig(cfg) == {}


def test_extract_runconfig_captures_precision_and_seed_and_batch():
    from mlops.hset_state import extract_runconfig
    cfg = {
        "trainer": {"precision": "bf16-mixed"},
        "seed_everything": 42,
        "data": {"init_args": {"batch_size": 32}},
    }
    out = extract_runconfig(cfg)
    assert out["trainer.precision"] == "bf16-mixed"
    assert out["seed_everything"] == 42
    assert out["data.init_args.batch_size"] == 32


def test_extract_runconfig_sorted_keys():
    """Output must be deterministically ordered for stable yaml diffs."""
    from mlops.hset_state import extract_runconfig
    cfg = {"model": {"weight_decay": 0.01, "lr": 3e-4, "beta1": 0.9}}
    out = extract_runconfig(cfg)
    assert list(out.keys()) == ["model.beta1", "model.lr", "model.weight_decay"]


# ── diff_runconfigs ────────────────────────────────────────────────────────

def test_diff_runconfigs_empty_when_same():
    from mlops.hset_state import diff_runconfigs
    a = {"model.lr": 3e-4}
    assert diff_runconfigs(a, dict(a)) == []


def test_diff_runconfigs_single_key_change():
    from mlops.hset_state import diff_runconfigs
    out = diff_runconfigs({"model.lr": 3e-4}, {"model.lr": 1e-4})
    assert out == [("model.lr", 1e-4, 3e-4)]


def test_diff_runconfigs_added_and_removed_keys():
    from mlops.hset_state import diff_runconfigs
    out = diff_runconfigs({"a": 1, "b": 2}, {"b": 2, "c": 3})
    assert out == [("a", None, 1), ("c", 3, None)]


def test_diff_runconfigs_none_handling():
    from mlops.hset_state import diff_runconfigs
    assert diff_runconfigs(None, None) == []
    assert diff_runconfigs(None, {"a": 1}) == [("a", 1, None)]
    assert diff_runconfigs({"a": 1}, None) == [("a", None, 1)]


# ── write/read_session_runconfig round-trip ───────────────────────────────

def test_runconfig_round_trip(tmp_path):
    from mlops.hset_state import (
        read_session_runconfig,
        write_session_runconfig,
    )
    cfg = {
        "model": {"dim": 512, "lr": 3e-4},
        "trainer": {"precision": "bf16-mixed", "max_epochs": 10},
        "seed_everything": 1,
    }
    write_session_runconfig(tmp_path, cfg)
    out = read_session_runconfig(tmp_path)
    assert out["model.lr"] == 3e-4
    assert out["trainer.precision"] == "bf16-mixed"
    assert out["seed_everything"] == 1
    assert "model.dim" not in out  # architecture stays out


def test_read_session_runconfig_missing_returns_none(tmp_path):
    from mlops.hset_state import read_session_runconfig
    assert read_session_runconfig(tmp_path) is None


def test_read_session_runconfig_unparseable_returns_none(tmp_path):
    from mlops.hset_state import read_session_runconfig
    (tmp_path / "runconfig.yaml").write_text("{not: valid: yaml")
    assert read_session_runconfig(tmp_path) is None


# ── parse_ckpt_step ────────────────────────────────────────────────────────

def test_parse_ckpt_step_from_filename(tmp_path):
    from mlops.hset_state import parse_ckpt_step
    p = tmp_path / "step=12345.ckpt"
    p.write_text("")
    assert parse_ckpt_step(p) == 12345


def test_parse_ckpt_step_follows_symlink(tmp_path):
    """`last.ckpt -> step=42.ckpt` resolves to step 42 — the common
    Lightning ModelCheckpoint convention."""
    from mlops.hset_state import parse_ckpt_step
    target = tmp_path / "step=42.ckpt"
    target.write_text("")
    last = tmp_path / "last.ckpt"
    last.symlink_to(target.name)
    assert parse_ckpt_step(last) == 42


def test_parse_ckpt_step_unknown_returns_none(tmp_path):
    from mlops.hset_state import parse_ckpt_step
    p = tmp_path / "checkpoint.pt"
    p.write_text("")
    assert parse_ckpt_step(p) is None
    assert parse_ckpt_step(None) is None


# ── latest_ckpt_step ───────────────────────────────────────────────────────

def test_latest_ckpt_step_empty_dir(tmp_path):
    from mlops.hset_state import latest_ckpt_step
    (tmp_path / "checkpoints").mkdir()
    assert latest_ckpt_step(tmp_path) is None


def test_latest_ckpt_step_no_dir(tmp_path):
    from mlops.hset_state import latest_ckpt_step
    assert latest_ckpt_step(tmp_path) is None


def test_latest_ckpt_step_picks_max(tmp_path):
    from mlops.hset_state import latest_ckpt_step
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in (10, 5000, 200, 99999):
        (ck / f"step={n}.ckpt").write_text("")
    # last.ckpt symlink shouldn't influence the count.
    (ck / "last.ckpt").symlink_to("step=99999.ckpt")
    assert latest_ckpt_step(tmp_path) == 99999


# ── detect_mode ────────────────────────────────────────────────────────────

def _v(name: str, tmp_path: Path) -> "VersionInfo":
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return VersionInfo(name=name, dir=d,
                       state_dict_hash="a"*64, model_file_hash="b"*64)


def _h(name: str, version: "VersionInfo") -> "HsetInfo":
    from mlops.hset_state import HsetInfo
    d = version.dir / name
    d.mkdir(exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    return HsetInfo(name=name, dir=d, config_hash="x")


def test_detect_mode_fresh_when_no_ckpt(tmp_path):
    from mlops.hset_state import Mode, detect_mode
    v = _v("v001", tmp_path)
    h = _h("h001", v)
    assert detect_mode(
        version=v, hset=h, ckpt_path=None,
        latest_version=v, latest_hset=h, latest_step=None,
    ) == Mode.FRESH


def test_detect_mode_resume_when_all_latest(tmp_path):
    from mlops.hset_state import Mode, detect_mode
    v = _v("v001", tmp_path)
    h = _h("h001", v)
    ck = h.dir / "checkpoints" / "step=100.ckpt"
    ck.write_text("")
    assert detect_mode(
        version=v, hset=h, ckpt_path=ck,
        latest_version=v, latest_hset=h, latest_step=100,
    ) == Mode.RESUME


def test_detect_mode_resume_when_explicit_step_matches_latest(tmp_path):
    """`--ckpt step=N.ckpt` where N is the latest = RESUME, not BACKTRACK.
    Spec's key insight: we look at what's actually loaded, not flag syntax."""
    from mlops.hset_state import Mode, detect_mode
    v = _v("v001", tmp_path)
    h = _h("h001", v)
    ck = h.dir / "checkpoints" / "step=796533.ckpt"
    ck.write_text("")
    assert detect_mode(
        version=v, hset=h, ckpt_path=ck,
        latest_version=v, latest_hset=h, latest_step=796533,
    ) == Mode.RESUME


def test_detect_mode_backtrack_when_older_step(tmp_path):
    from mlops.hset_state import Mode, detect_mode
    v = _v("v001", tmp_path)
    h = _h("h001", v)
    ck = h.dir / "checkpoints" / "step=100.ckpt"
    ck.write_text("")
    assert detect_mode(
        version=v, hset=h, ckpt_path=ck,
        latest_version=v, latest_hset=h, latest_step=999,
    ) == Mode.BACKTRACK


def test_detect_mode_backtrack_when_older_hset(tmp_path):
    from mlops.hset_state import Mode, detect_mode
    v = _v("v001", tmp_path)
    h1 = _h("h001", v)
    h3 = _h("h003", v)
    ck = h1.dir / "checkpoints" / "step=50.ckpt"
    ck.write_text("")
    assert detect_mode(
        version=v, hset=h1, ckpt_path=ck,
        latest_version=v, latest_hset=h3, latest_step=50,
    ) == Mode.BACKTRACK


def test_detect_mode_backtrack_when_older_version(tmp_path):
    from mlops.hset_state import Mode, detect_mode
    v1 = _v("v001", tmp_path)
    v3 = _v("v003", tmp_path)
    h = _h("h001", v1)
    ck = h.dir / "checkpoints" / "step=50.ckpt"
    ck.write_text("")
    assert detect_mode(
        version=v1, hset=h, ckpt_path=ck,
        latest_version=v3, latest_hset=h, latest_step=50,
    ) == Mode.BACKTRACK
