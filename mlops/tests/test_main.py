"""Unit tests for mlops.main.

`build_lightning_cli` is exercised indirectly by integration tests in a
later step; it requires a real datamodule wired up to make end-to-end
testing meaningful.
"""
import sys
import textwrap
from pathlib import Path

import pytest

from mlops.main import (
    LIGHTNING_PREFIXES,  # noqa: F401  (re-exported for symmetry)
    _is_lightning_flag,
    build_cascade_args,
    discover_model_class,
    main,
    split_argv,
)
from mlops.hset_state import Ft4Args, Ft4UserError


@pytest.fixture(autouse=True)
def _clear_my_model_from_sys_modules():
    # Many tests below register a transient `my_model` module from a tmp_path
    # file. Without cleanup, the next test sees a stale entry pointing at a
    # deleted tmp dir, and the shadowing guard fires.
    sys.modules.pop("my_model", None)
    yield
    sys.modules.pop("my_model", None)


# ── split_argv ─────────────────────────────────────────────────────────────

def test_split_argv_plain():
    ft4, lit = split_argv(["run", "model.py"])
    assert ft4 == ["run", "model.py"]
    assert lit == []


def test_split_argv_ft4_flag():
    ft4, lit = split_argv(["run", "model.py", "--hset", "h007"])
    assert ft4 == ["run", "model.py", "--hset", "h007"]
    assert lit == []


def test_split_argv_lightning_eq_syntax():
    ft4, lit = split_argv(["run", "model.py", "--trainer.max_epochs=4"])
    assert ft4 == ["run", "model.py"]
    assert lit == ["--trainer.max_epochs=4"]


def test_split_argv_lightning_space_syntax():
    ft4, lit = split_argv(["run", "model.py", "--trainer.max_epochs", "4"])
    assert ft4 == ["run", "model.py"]
    assert lit == ["--trainer.max_epochs", "4"]


def test_split_argv_mixed():
    ft4, lit = split_argv([
        "run", "model.py", "--hset", "h007",
        "--trainer.max_epochs=4", "--model.dim", "512",
        "--ckpt", "last",
    ])
    assert ft4 == ["run", "model.py", "--hset", "h007", "--ckpt", "last"]
    assert lit == ["--trainer.max_epochs=4", "--model.dim", "512"]


def test_split_argv_config_with_space():
    ft4, lit = split_argv(["run", "model.py", "--config", "extra.yaml"])
    assert ft4 == ["run", "model.py"]
    assert lit == ["--config", "extra.yaml"]


def test_split_argv_seed_everything():
    ft4, lit = split_argv(["run", "model.py", "--seed_everything=42"])
    assert ft4 == ["run", "model.py"]
    assert lit == ["--seed_everything=42"]


def test_split_argv_no_value_for_lightning_flag_at_end():
    # Bare flag at end with no value; we don't grab past the end.
    ft4, lit = split_argv(["run", "model.py", "--config"])
    assert ft4 == ["run", "model.py"]
    assert lit == ["--config"]


# ── _is_lightning_flag ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "tok,expected",
    [
        # Lightning prefixes.
        ("--model.dim", True),
        ("--model.dim=512", True),
        ("--data.batch_size", True),
        ("--trainer.max_epochs", True),
        ("--trainer.max_epochs=4", True),
        ("--config", True),
        ("--config=foo.yaml", True),
        ("--seed_everything", True),
        ("--seed_everything=42", True),
        ("--optimizer", True),
        ("--optimizer.lr", True),
        ("--lr_scheduler", True),
        # ft4-side flags.
        ("--hset", False),
        ("--ckpt", False),
        ("--model-class", False),
        ("--vdesc", False),
        ("--new-version", False),
        ("--new-hset", False),
        ("--rdesc", False),
        # Positionals.
        ("run", False),
        ("model.py", False),
    ],
)
def test_is_lightning_flag(tok, expected):
    assert _is_lightning_flag(tok) == expected


# ── argparse + Ft4Args ─────────────────────────────────────────────────────

def test_main_missing_model_file_returns_2(tmp_path, capsys):
    rc = main(["train", str(tmp_path / "does_not_exist.py")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_main_validate_subcommand_is_gone(tmp_path):
    """`validate` was dropped — argparse should reject it as an unknown
    subcommand. (`train`, `generate`, `show`, `list` are wired.)"""
    f = tmp_path / "m.py"
    f.write_text("# placeholder\n")
    with pytest.raises(SystemExit):
        main(["validate", str(f)])


def test_main_unknown_subcommand_fails(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("# placeholder\n")
    # argparse rejects with SystemExit.
    with pytest.raises(SystemExit):
        main(["bogus", str(f)])


# ── build_cascade_args ─────────────────────────────────────────────────────

def _args_for(model_file: Path, lightning_args: list[str] | None = None) -> Ft4Args:
    return Ft4Args(
        subcommand="train",
        model_file=model_file,
        lightning_args=lightning_args or [],
    )


def test_cascade_args_order(tmp_path):
    ft4_yaml = tmp_path / "ft4.yaml"
    ft4_yaml.write_text("trainer: {}\n")
    models_defaults = tmp_path / "models_defaults.yaml"
    models_defaults.write_text("trainer:\n  precision: 32-true\n")
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  dim: 8\n")
    cascade_paths = [ft4_yaml, models_defaults, model_yaml]

    cascade = build_cascade_args(
        _args_for(model_file, ["--trainer.max_epochs=4"]),
        cascade_paths=cascade_paths,
    )
    assert cascade == [
        "--config", str(ft4_yaml),
        "--config", str(models_defaults),
        "--config", str(model_yaml),
        "--trainer.max_epochs=4",
    ]


def test_cascade_args_skips_missing_files(tmp_path):
    ft4_yaml = tmp_path / "ft4.yaml"
    ft4_yaml.write_text("trainer: {}\n")
    missing = tmp_path / "nope.yaml"
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    cascade = build_cascade_args(
        _args_for(model_file, ["--trainer.max_epochs=4"]),
        cascade_paths=[ft4_yaml, missing],
    )
    assert cascade == [
        "--config", str(ft4_yaml),
        "--trainer.max_epochs=4",
    ]


def test_cascade_args_empty_layers_uses_only_cli(tmp_path):
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    cascade = build_cascade_args(
        _args_for(model_file, ["--trainer.max_epochs=4"]),
        cascade_paths=[],
    )
    assert cascade == ["--trainer.max_epochs=4"]


def test_default_cascade_paths_layout(tmp_path):
    """The default cascade lists six paths in the documented order, so a
    student reading the help can grep for them on disk."""
    from mlops.main import _default_cascade_paths
    model_file = tmp_path / "submodels" / "mymodel.py"
    paths = _default_cascade_paths(model_file)
    names = [p.name for p in paths]
    assert names == [
        "ft4.yaml", "ft4.local.yaml",
        "defaults.yaml", "defaults.local.yaml",
        "mymodel.yaml", "mymodel.local.yaml",
    ]
    # The shipped defaults.yaml must resolve to a file that exists — otherwise
    # build_cascade_args silently skips it (the `if path.exists()` guard) and
    # every key in it is dropped from the merged config.
    defaults_path = next(p for p in paths if p.name == "defaults.yaml")
    assert defaults_path.exists(), (
        f"defaults.yaml not found at {defaults_path}; cascade would silently drop it"
    )


# ── --hset auto-load ──────────────────────────────────────────────────────

def _make_hset_dir(runs_root: Path, stem: str, version: str, hset: str,
                   config_yaml_text: str = "model:\n  dim: 64\n") -> Path:
    d = runs_root / stem / version / hset
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(config_yaml_text)
    return d


def test_hset_autoload_substitutes_model_yaml_in_cascade(tmp_path, capsys):
    """When --hset hNNN is given and the hset's config.yaml exists, the
    cascade drops the model.yaml/model.local.yaml layers and appends the
    hset's saved config instead. CLI args still win."""
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  dim: 8\n")
    ft4_yaml = tmp_path / "ft4.yaml"
    ft4_yaml.write_text("trainer: {}\n")

    runs_root = tmp_path / "runs"
    hset_dir = _make_hset_dir(runs_root, "m", "v001", "h003")
    hset_config = hset_dir / "config.yaml"

    args = _args_for(model_file, ["--trainer.max_epochs=4"])
    args = Ft4Args(**{**args.__dict__, "hset": "h003"})

    cascade = build_cascade_args(
        args, cascade_paths=[ft4_yaml, model_yaml], runs_root=runs_root,
    )
    # model_yaml is dropped (sits under model_file.parent), hset_config appended.
    assert cascade == [
        "--config", str(ft4_yaml),
        "--config", str(hset_config),
        "--trainer.max_epochs=4",
    ]
    assert "loading hparams from" in capsys.readouterr().err


def test_hset_autoload_skipped_when_new_hset_set(tmp_path):
    """--new-hset means 'fork a new hset', so don't substitute even if the
    named hset exists."""
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  dim: 8\n")

    runs_root = tmp_path / "runs"
    _make_hset_dir(runs_root, "m", "v001", "h003")

    args = _args_for(model_file, [])
    args = Ft4Args(**{**args.__dict__, "hset": "h003", "new_hset": True})

    cascade = build_cascade_args(
        args, cascade_paths=[model_yaml], runs_root=runs_root,
    )
    assert cascade == ["--config", str(model_yaml)]


def test_hset_autoload_ambiguous_falls_through(tmp_path):
    """If the same hset name exists under multiple versions and no --version
    pin is given, don't guess — fall through to the normal cascade."""
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    model_yaml = tmp_path / "m.yaml"
    model_yaml.write_text("model:\n  dim: 8\n")

    runs_root = tmp_path / "runs"
    _make_hset_dir(runs_root, "m", "v001", "h003")
    _make_hset_dir(runs_root, "m", "v002", "h003")

    args = _args_for(model_file, [])
    args = Ft4Args(**{**args.__dict__, "hset": "h003"})

    cascade = build_cascade_args(
        args, cascade_paths=[model_yaml], runs_root=runs_root,
    )
    # No substitution — model.yaml kept; resolve_hset will handle the error.
    assert cascade == ["--config", str(model_yaml)]


def test_hset_autoload_uses_version_pin_when_ambiguous(tmp_path):
    """With --version, the lookup is unambiguous even if other versions
    contain the same hset."""
    model_file = tmp_path / "m.py"
    model_file.write_text("# m")
    runs_root = tmp_path / "runs"
    _make_hset_dir(runs_root, "m", "v001", "h003", "model:\n  dim: 8\n")
    pinned = _make_hset_dir(runs_root, "m", "v002", "h003", "model:\n  dim: 16\n")

    args = _args_for(model_file, [])
    args = Ft4Args(**{**args.__dict__, "hset": "h003", "version": "v002"})

    cascade = build_cascade_args(args, cascade_paths=[], runs_root=runs_root)
    assert cascade == ["--config", str(pinned / "config.yaml")]


# ── discover_model_class ───────────────────────────────────────────────────

_MODULE_WITH_ONE = textwrap.dedent("""
    import lightning.pytorch as pl

    class MyTransformer(pl.LightningModule):
        def __init__(self):
            super().__init__()
""")

_MODULE_WITH_TWO = textwrap.dedent("""
    import lightning.pytorch as pl

    class A(pl.LightningModule):
        def __init__(self):
            super().__init__()

    class B(pl.LightningModule):
        def __init__(self):
            super().__init__()
""")

_MODULE_WITH_IMPORTED_LM = textwrap.dedent("""
    # User imports LightningModule but doesn't define a subclass here.
    from lightning.pytorch import LightningModule
""")

_MODULE_EMPTY = "# empty\n"


def test_discover_model_class_single(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_WITH_ONE)
    cls = discover_model_class(f)
    assert cls.__name__ == "MyTransformer"


def test_discover_model_class_multiple_errors(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_WITH_TWO)
    with pytest.raises(Ft4UserError, match="multiple"):
        discover_model_class(f)


def test_discover_model_class_multiple_with_explicit(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_WITH_TWO)
    assert discover_model_class(f, explicit="A").__name__ == "A"
    assert discover_model_class(f, explicit="B").__name__ == "B"


def test_discover_model_class_explicit_missing_name(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_WITH_ONE)
    with pytest.raises(Ft4UserError, match="not found"):
        discover_model_class(f, explicit="DoesNotExist")


def test_discover_model_class_explicit_not_lightning(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(textwrap.dedent("""
        class NotALightningModule:
            pass
    """))
    with pytest.raises(Ft4UserError, match="not a LightningModule"):
        discover_model_class(f, explicit="NotALightningModule")


def test_discover_model_class_none_defined(tmp_path):
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_EMPTY)
    with pytest.raises(Ft4UserError, match="no LightningModule"):
        discover_model_class(f)


def test_discover_model_class_only_imported_not_counted(tmp_path):
    """Importing LightningModule shouldn't make the file 'have' one."""
    f = tmp_path / "my_model.py"
    f.write_text(_MODULE_WITH_IMPORTED_LM)
    with pytest.raises(Ft4UserError, match="no LightningModule"):
        discover_model_class(f)


def test_discover_model_class_propagates_import_error(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("this is not valid python !!!\n")
    with pytest.raises(Ft4UserError, match="failed to import"):
        discover_model_class(f)


def test_discover_model_class_refuses_stdlib_collision(tmp_path):
    """Naming the file `sys.py` would shadow the stdlib `sys` module."""
    f = tmp_path / "sys.py"
    f.write_text(_MODULE_EMPTY)
    with pytest.raises(Ft4UserError, match="standard library"):
        discover_model_class(f)


def test_discover_model_class_refuses_installed_pkg_collision(tmp_path):
    """Naming the file `torch.py` (or any already-imported package name)
    would silently break every subsequent `import torch` for the process."""
    f = tmp_path / "torch.py"
    f.write_text(_MODULE_EMPTY)
    with pytest.raises(Ft4UserError, match="collides with the already-imported"):
        discover_model_class(f)


# ── Help system ────────────────────────────────────────────────────────────

def test_top_help_mentions_subcommands_and_concepts(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ft4 train" in out
    assert "version" in out
    assert "hset" in out
    assert "session" in out
    # Student-MLops workflow commands should be discoverable.
    assert "ft4 generate" in out
    assert "ft4 list" in out
    # The cascade hint must list the config files (last wins).
    assert "defaults.yaml" in out


def test_train_help_lists_hset_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--hset" in out
    assert "--hdesc" in out
    assert "--lightning-help" in out
    assert "--new-hset" in out
    assert "--rdesc" in out


def test_train_args_parses_rdesc_and_new_hset(tmp_path):
    """--rdesc and --new-hset round-trip through _build_ft4_args."""
    from mlops.main import _build_parser, _build_ft4_args
    model_file = tmp_path / "m.py"
    model_file.write_text("# placeholder\n")
    parsed = _build_parser().parse_args(
        ["train", str(model_file), "--rdesc", "warmup + cosine", "--new-hset"]
    )
    ft4_args = _build_ft4_args(parsed, lightning_argv=[])
    assert ft4_args.rdesc == "warmup + cosine"
    assert ft4_args.new_hset is True
    assert ft4_args.new_version is False  # default unchanged


# ── _check_cascade_for_todos ───────────────────────────────────────────────

def test_check_cascade_for_todos_raises_with_listing(tmp_path):
    """A yaml carrying !TODO placeholders should produce a Ft4UserError that
    lists every unfilled line, so the student sees them all at once instead
    of fixing one, running again, hitting the next."""
    from mlops.main import _check_cascade_for_todos
    bad = tmp_path / "model.yaml"
    bad.write_text(
        "model:\n"
        '  dim: !TODO "TODO-LAB unit1.lab2"\n'
        '  depth: !TODO "TODO-LAB unit1.lab2"\n'
        "  lr: 3.0e-4\n"
    )
    with pytest.raises(Ft4UserError) as exc_info:
        _check_cascade_for_todos(["--config", str(bad)])
    msg = str(exc_info.value)
    # Every unfilled line is shown — not just the first.
    assert msg.count("TODO-LAB unit1.lab2") == 2
    assert "dim" in msg and "depth" in msg
    # The file path is shown so the student knows where to edit.
    assert str(bad) in msg


def test_check_cascade_for_todos_silent_when_clean(tmp_path):
    """A cascade with no !TODO placeholders should not raise."""
    from mlops.main import _check_cascade_for_todos
    good = tmp_path / "model.yaml"
    good.write_text("model:\n  dim: 64\n  depth: 4\n")
    _check_cascade_for_todos(["--config", str(good), "--trainer.max_epochs=4"])


def test_check_cascade_for_todos_skips_missing_files(tmp_path):
    """build_cascade_args filters non-existent files, but defensively the
    pre-scan must not crash on a missing path either."""
    from mlops.main import _check_cascade_for_todos
    _check_cascade_for_todos(["--config", str(tmp_path / "nope.yaml")])


def test_check_cascade_for_todos_ignores_comments(tmp_path):
    """A `!TODO` token inside a YAML comment is not an unfilled placeholder
    (e.g. a comment describing how the system works)."""
    from mlops.main import _check_cascade_for_todos
    p = tmp_path / "model.yaml"
    p.write_text(
        '# Use !TODO "..." to mark unfilled hparams.\n'
        "model:\n"
        "  dim: 64\n"
    )
    _check_cascade_for_todos(["--config", str(p)])


def test_check_cascade_for_todos_scans_all_layers(tmp_path):
    """Multiple --config args means multiple files to scan; an unfilled
    !TODO in any of them must surface, regardless of which layer it's in."""
    from mlops.main import _check_cascade_for_todos
    clean = tmp_path / "ft4.yaml"
    clean.write_text("trainer: {}\n")
    dirty = tmp_path / "model.yaml"
    dirty.write_text('model:\n  dim: !TODO "TODO-LAB unit1.lab2"\n')
    with pytest.raises(Ft4UserError) as exc_info:
        _check_cascade_for_todos([
            "--config", str(clean),
            "--config", str(dirty),
            "--trainer.max_epochs=4",
        ])
    assert "TODO-LAB unit1.lab2" in str(exc_info.value)
    assert str(dirty) in str(exc_info.value)


def test_check_cascade_for_todos_accepts_both_tag_forms(tmp_path):
    """!TODO and !TODO-LAB should both be flagged."""
    from mlops.main import _check_cascade_for_todos
    p = tmp_path / "model.yaml"
    p.write_text(
        "model:\n"
        '  a: !TODO "unit1.lab2"\n'
        '  b: !TODO-LAB "unit1.lab3"\n'
    )
    with pytest.raises(Ft4UserError) as exc_info:
        _check_cascade_for_todos(["--config", str(p)])
    msg = str(exc_info.value)
    assert "2 unfilled" in msg
    assert "unit1.lab2" in msg and "unit1.lab3" in msg


def test_check_cascade_for_todos_no_false_match_on_similar_tags(tmp_path):
    """A tag named `!TODOX` or `!TODO-LABEL` is a different tag (would fail
    YAML parse for other reasons). The pre-scan must not flag those —
    catching them would mask the real error and confuse the student."""
    from mlops.main import _check_cascade_for_todos
    p = tmp_path / "model.yaml"
    # Use quoted strings so the contents are valid YAML; the regex is what
    # we're testing.
    p.write_text(
        "model:\n"
        '  note1: "this mentions !TODOSOMETHING"\n'
        '  note2: "this mentions !TODO-LABEL"\n'
    )
    _check_cascade_for_todos(["--config", str(p)])


def test_check_cascade_for_todos_handles_equals_config_syntax(tmp_path):
    """jsonargparse accepts both `--config foo.yaml` and `--config=foo.yaml`;
    the pre-scan must scan files referenced either way."""
    from mlops.main import _check_cascade_for_todos
    bad = tmp_path / "model.yaml"
    bad.write_text('model:\n  dim: !TODO "TODO-LAB unit1.lab2"\n')
    with pytest.raises(Ft4UserError) as exc_info:
        _check_cascade_for_todos([f"--config={bad}", "--trainer.max_epochs=4"])
    assert "TODO-LAB unit1.lab2" in str(exc_info.value)


def test_check_cascade_for_todos_dedupes_repeated_paths(tmp_path):
    """If the same yaml path appears twice in the cascade, each !TODO line
    should appear once in the error — not duplicated."""
    from mlops.main import _check_cascade_for_todos
    bad = tmp_path / "model.yaml"
    bad.write_text('model:\n  dim: !TODO "TODO-LAB unit1.lab2"\n')
    with pytest.raises(Ft4UserError) as exc_info:
        _check_cascade_for_todos([
            "--config", str(bad),
            "--config", str(bad),
        ])
    msg = str(exc_info.value)
    assert msg.count("TODO-LAB unit1.lab2") == 1
    assert "1 unfilled" in msg


def test_check_cascade_for_todos_skips_when_help_requested(tmp_path):
    """`ft4 train m.py --lightning-help` should still print Lightning's help
    even when the yaml has unfilled !TODOs — students need to be able to
    discover what flags exist before they can fill anything in."""
    from mlops.main import _check_cascade_for_todos
    bad = tmp_path / "model.yaml"
    bad.write_text('model:\n  dim: !TODO "TODO-LAB unit1.lab2"\n')
    # Must not raise — --help signals an introspection request, not a real run.
    _check_cascade_for_todos(["--config", str(bad), "--help"])
    _check_cascade_for_todos(["--config", str(bad), "-h"])
