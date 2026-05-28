"""Main entry point for ft4 CLI.

Walks through:
  1. Split argv into ft4 flags and Lightning flags.
  2. argparse the ft4 side into an Ft4Args dataclass.
  3. Dispatch to the subcommand handler.

The pieces this module exposes for downstream steps to compose:
  - split_argv:            argv → (ft4_argv, lightning_argv)
  - discover_model_class:  resolve a LightningModule from a file
  - build_cascade_args:    build the --config layering for LightningCLI
  - build_lightning_cli:   wrap LightningCLI construction (run=False, no
                           SaveConfigCallback)

`train` dispatches into mlops.run (smart-flow runner); `generate`, `show`,
and `list` each have their own module under mlops.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import threading
import random
from pathlib import Path

from mlops.todo_yaml import register as _register_todo_yaml
from mlops.hset_state import Ft4Args, Ft4UserError


# Token prefixes that route a flag into the Lightning bucket. Anything not
# matching falls through to ft4's argparse. Ordering doesn't matter; each is
# checked independently.
LIGHTNING_PREFIXES: tuple[str, ...] = (
    "--model.", "--data.", "--trainer.",
    "--config", "--seed_everything",
    "--optimizer", "--lr_scheduler",
)


# ── Entry points ───────────────────────────────────────────────────────────

def _install_dataloader_shutdown_filter() -> None:
    """Suppress benign exceptions from PyTorch DataLoader background threads
    during Ctrl-C teardown.

    The pin-memory daemon thread and worker threads can be mid-recv on a
    Unix socket when the DataLoader's workers are torn down, producing a
    ConnectionResetError / EOFError / BrokenPipeError that does not affect
    program exit but prints a noisy traceback. Filter only those.
    """
    prior = threading.excepthook
    benign_types = (ConnectionResetError, EOFError, BrokenPipeError)
    benign_files = ("pin_memory.py", "worker.py")

    def hook(args):
        exc = args.exc_value
        if isinstance(exc, benign_types):
            tb = args.exc_traceback
            while tb is not None:
                fname = tb.tb_frame.f_code.co_filename
                if fname.endswith(benign_files) and "torch" in fname:
                    return
                tb = tb.tb_next
        prior(args)

    threading.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    """Top-level entry. Returns exit code (0=ok, 2=user error)."""
    _register_todo_yaml()
    _install_dataloader_shutdown_filter()
    if argv is None:
        argv = sys.argv[1:]
    try:
        ft4_argv, lightning_argv = split_argv(argv)
        parsed = _build_parser().parse_args(ft4_argv)
        if getattr(parsed, "lightning_help", False):
            return _show_lightning_help(parsed)
        ft4_args = _build_ft4_args(parsed, lightning_argv)
        return _dispatch(ft4_args)
    except Ft4UserError as e:
        print(f"\n* ERROR: {e}", file=sys.stderr)
        return 2


def _show_lightning_help(parsed: argparse.Namespace) -> int:
    """Print LightningCLI's --help for the given model and exit. Lists every
    --model.X / --data.X / --trainer.X flag jsonargparse derived for this
    specific model class."""
    model_file: Path = parsed.model_file
    if not model_file.exists():
        raise Ft4UserError(f"model file not found: {model_file}")
    model_class = discover_model_class(model_file.resolve(), parsed.model_class)
    ft4_args = Ft4Args(
        subcommand=parsed.subcommand,
        model_file=model_file.resolve(),
        model_class=parsed.model_class,
        lightning_args=["--help"],
    )
    # build_lightning_cli with --help in args causes jsonargparse to print
    # help and SystemExit(0). We catch and translate.
    try:
        build_lightning_cli(ft4_args, model_class)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    return 0


# ── argv splitting ─────────────────────────────────────────────────────────

def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Send Lightning flags to one list and everything else (subcommand,
    model file, ft4 flags) to the other. Handles both `--flag=value` and
    `--flag value` syntax.

    Examples (see test_main.py):
      ["run", "m.py", "--hset", "h7"]                      → ft4, []
      ["run", "m.py", "--trainer.max_epochs=4"]            → [..run..], [..lit..]
      ["run", "m.py", "--trainer.max_epochs", "4"]         → consumes both as lit
    """
    ft4: list[str] = []
    lit: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if _is_lightning_flag(tok):
            lit.append(tok)
            # Space-separated value: pull the next token too.
            if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                lit.append(argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            ft4.append(tok)
            i += 1
    return ft4, lit


def _is_lightning_flag(tok: str) -> bool:
    if not tok.startswith("--"):
        return False
    for p in LIGHTNING_PREFIXES:
        if p.endswith("."):
            # Namespace prefix (e.g., --model., --trainer.).
            if tok.startswith(p):
                return True
        else:
            # Exact match or = / . follow-on (e.g., --config, --config=foo,
            # --optimizer.lr=...).
            if tok == p or tok.startswith(p + "=") or tok.startswith(p + "."):
                return True
    return False


# ── ft4-arg parsing ────────────────────────────────────────────────────────

_TOP_DESCRIPTION = (
    "ft4 — train, resume, generate, and compare runs without losing track of versions.\n"
    "Wraps PyTorch Lightning so you can focus on the model, not the bookkeeping."
)

_TOP_EPILOG = """\
[bold]Concepts[/bold]
  [yellow]version[/yellow]   (vNNN)  forks when your model's [bold]code[/bold] changes
  [yellow]hset[/yellow]      (hNNN)  forks when [bold]hyperparameters[/bold] change
  [yellow]runconfig[/yellow]         training recipe (lr, optimizer, precision, dropout)
  [yellow]session[/yellow]   (sNNN)  one training run (start -> finish or interrupt)

[bold]Config[/bold]
  Configure via [cyan]defaults.yaml[/cyan], [cyan]<model name>.yaml[/cyan], or CLI flags

[bold]Common workflows[/bold]
  [cyan]ft4 train path/to/model.py[/cyan]                                    train (or resume) the latest hset
  [cyan]ft4 train path/to/model.py --model.dim=256 --model.depth=8[/cyan]    experiment with hyperparameters
  [cyan]ft4 train path/to/model.py --model.lr=1e-3[/cyan]                    change lr
  [cyan]ft4 train path/to/model.py --compile --trainer.max_epochs=30[/cyan]  compile (2x speed boost) and train further
  [cyan]ft4 list path/to/model.py[/cyan]                                     compare all hsets for this model
  [cyan]ft4 show path/to/model.py --hset h003[/cyan]                         inspect an hset without training
  [cyan]ft4 train path/to/model.py --hset h003[/cyan]                        resume an old hset (loads its saved hparams)
  [cyan]ft4 generate path/to/model.py --prompt "Sammy jumped"[/cyan]         generate, using the latest checkpoint

To see all flags (e.g. --trainer.X / --data.X), including hyperparameters specific to your model (--model.X), do
  [cyan]ft4 train <model.py> --lightning-help[/cyan].
"""

_SUB_EPILOG = """\
[bold]Examples[/bold]
  [cyan]ft4 train model.py[/cyan]                                  train or resume
  [cyan]ft4 train model.py --hdesc "smaller LR"[/cyan]             label this hset for future-you
  [cyan]ft4 train model.py --trainer.max_epochs=20[/cyan]          add more epochs
  [cyan]ft4 train model.py --hset h003[/cyan]                      resume an old hset (its saved hparams are reloaded)
  [cyan]ft4 generate model.py --prompt "Once upon"[/cyan]          sample from the checkpoint
  [cyan]ft4 list model.py --by-perf[/cyan]                         list hsets ranked by val loss
"""


def _formatter_class():
    """RawDescriptionRichHelpFormatter renders colored, sectioned help and
    preserves newlines/markup in description and epilog blocks. Falls back to
    argparse's raw formatter if rich-argparse isn't installed (it's a direct
    dep, so the import should always succeed in practice)."""
    try:
        from rich_argparse import RawDescriptionRichHelpFormatter
        return RawDescriptionRichHelpFormatter
    except ImportError:
        return argparse.RawDescriptionHelpFormatter


def _build_parser() -> argparse.ArgumentParser:
    formatter_class = _formatter_class()
    parser = argparse.ArgumentParser(
        prog="ft4",
        description=_TOP_DESCRIPTION,
        epilog=_TOP_EPILOG,
        formatter_class=formatter_class,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "model_file", type=Path,
            help="Path to the model .py file (defines a LightningModule).",
        )
        selection = sp.add_argument_group("selection")
        selection.add_argument(
            "--hset", default=None, metavar="hNNN",
            help="Pin to an existing hset (e.g. h002). For write ops, hparam hash must match.",
        )
        selection.add_argument(
            "--version", default=None, metavar="vNNN",
            help="Pin to an existing model version (e.g. v002). Errors if architecture differs.",
        )
        selection.add_argument(
            "--new-version", action="store_true", dest="new_version",
            help="Force creation of a new model version, even if the latest one matches.",
        )
        selection.add_argument(
            "--new-hset", action="store_true", dest="new_hset",
            help="Force creation of a new hset within the version, even if the current "
                 "config hashes to match an existing one (e.g. to branch off an old ckpt).",
        )
        selection.add_argument(
            "--ckpt", default="last", metavar="PATH|last",
            help="Checkpoint to resume from. 'last' (default) loads <hset>/checkpoints/last.ckpt.",
        )
        selection.add_argument(
            "--model-class", dest="model_class", default=None, metavar="ClassName",
            help="Disambiguate when the model file defines multiple LightningModule subclasses.",
        )
        descriptions = sp.add_argument_group("descriptions")
        descriptions.add_argument(
            "--vdesc", default=None, metavar="TEXT",
            help="Set/update human-readable description for this model version.",
        )
        descriptions.add_argument(
            "--hdesc", default=None, metavar="TEXT",
            help="Set/update human-readable description for this hset.",
        )
        descriptions.add_argument(
            "--rdesc", default=None, metavar="TEXT",
            help="Human-readable description of this session's runconfig "
                 "(training recipe: LR, schedule, precision). Stored in sessions.jsonl.",
        )
        behavior = sp.add_argument_group("behavior")
        behavior.add_argument(
            "--compile", action="store_true",
            help="Compile the model with torch.compile before training.",
        )
        behavior.add_argument(
            "--prompt", action="append", default=None, metavar="TEXT",
            help="Sample prompt for generative models. Repeatable.",
        )
        behavior.add_argument(
            "--wandb", action="store_true",
            help="Enable wandb experiment tracking. Errors if wandb is not "
                 "installed or you are not logged in.",
        )
        behavior.add_argument(
            "--lightning-help", action="store_true", dest="lightning_help",
            help="Print Lightning's help (--model.X, --data.X, --trainer.X flags) for this model and exit.",
        )

    subcommand_help = {
        "train": "Train, or resume training, the model; forks a new "
                 "version on model code change and a new hset on "
                 "hyperparameter change.",
        "generate": "Run generation from a trained checkpoint. Prints "
                    "prompt+completion pairs to stdout.",
        "show": "Print the briefing for one hset without training.",
    }
    for cmd in ("train", "generate", "show"):
        sp = sub.add_parser(
            cmd,
            help=subcommand_help[cmd],
            description=subcommand_help[cmd],
            epilog=_SUB_EPILOG,
            formatter_class=formatter_class,
        )
        common(sp)
        if cmd == "generate":
            _add_generate_flags(sp)

    # `ft4 list` — read-only directory walk; doesn't share `common()` flags
    # since most don't apply (no --ckpt, --prompt, --wandb, etc.).
    list_sp = sub.add_parser(
        "list",
        help="Tabular summary of versions × hsets for one model.",
        description="Tabular summary of versions × hsets for one model.",
        epilog=_SUB_EPILOG,
        formatter_class=formatter_class,
    )
    list_sp.add_argument(
        "model_file", type=Path,
        help="Path to the model .py file (its filesystem stem is used to find runs/<stem>/).",
    )
    list_sp.add_argument(
        "--version", default=None, metavar="vNNN",
        help="Filter to one model version.",
    )
    sort_group = list_sp.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--by-time", dest="sort_by", action="store_const", const="time",
        help="Sort by most-recent activity (default).",
    )
    sort_group.add_argument(
        "--by-perf", dest="sort_by", action="store_const", const="perf",
        help="Sort by best validation metric (smallest first).",
    )
    list_sp.set_defaults(sort_by="time")
    list_sp.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Show only the top N rows after sorting.",
    )

    return parser


def _add_generate_flags(sp: argparse.ArgumentParser) -> None:
    """Sampling controls for `ft4 generate`. None defaults defer to LangGen's
    own defaults; `--max-tokens` falls back to 2048 in `run_generate`."""
    g = sp.add_argument_group("generation")
    g.add_argument(
        "--temperature", type=float, default=None, metavar="T",
        help="Sampling temperature (LangGen default 0.7).",
    )
    g.add_argument(
        "--min-p", dest="min_p", type=float, default=None, metavar="P",
        help="Min-p sampling cutoff (LangGen default 0.2).",
    )
    g.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None, metavar="N",
        help="Cap generation at N tokens (default 2048). LangGen stops "
             "earlier on the tokenizer's end-of-text token.",
    )
    g.add_argument(
        "--force", action="store_true",
        help="Skip the 'not trained enough' warning.",
    )


def _build_ft4_args(parsed: argparse.Namespace, lightning_argv: list[str]) -> Ft4Args:
    """Build Ft4Args from a parsed namespace. Uses getattr-with-default for
    fields not present on every subcommand's parser (e.g., `list` doesn't
    define --hset / --ckpt / --prompt)."""
    model_file: Path = parsed.model_file
    if not model_file.exists():
        raise Ft4UserError(f"model file not found: {model_file}")
    return Ft4Args(
        subcommand=parsed.subcommand,
        model_file=model_file.resolve(),
        model_class=getattr(parsed, "model_class", None),
        hset=getattr(parsed, "hset", None),
        version=getattr(parsed, "version", None),
        ckpt=getattr(parsed, "ckpt", "last"),
        new_version=getattr(parsed, "new_version", False),
        new_hset=getattr(parsed, "new_hset", False),
        vdesc=getattr(parsed, "vdesc", None),
        hdesc=getattr(parsed, "hdesc", None),
        rdesc=getattr(parsed, "rdesc", None),
        prompt=getattr(parsed, "prompt", None),
        compile=getattr(parsed, "compile", False),
        wandb=getattr(parsed, "wandb", False),
        temperature=getattr(parsed, "temperature", None),
        min_p=getattr(parsed, "min_p", None),
        max_tokens=getattr(parsed, "max_tokens", None),
        force=getattr(parsed, "force", False),
        sort_by=getattr(parsed, "sort_by", "time"),
        limit=getattr(parsed, "limit", None),
        lightning_args=lightning_argv,
    )


# ── Dispatch ───────────────────────────────────────────────────────────────

def _dispatch(args: Ft4Args) -> int:
    if args.subcommand == "train":
        # The smart-flow runner. (Module name `run` is the historical implementation
        # symbol — a follow-up will rename it to match the subcommand.)
        from mlops.run import run
        return run(args)
    if args.subcommand == "list":
        from mlops.list_cmd import run_list
        return run_list(args)
    if args.subcommand == "show":
        from mlops.show import run_show
        return run_show(args)
    if args.subcommand == "generate":
        from mlops.generate import run_generate
        return run_generate(args)
    raise Ft4UserError(f"unknown subcommand: {args.subcommand}")


# ── Model-class discovery ──────────────────────────────────────────────────

def discover_model_class(model_file: Path, explicit: str | None = None):
    """Import `model_file` as a module and return its LightningModule subclass.

    With `explicit`, return the named class (must subclass LightningModule).
    Otherwise, return the unique LightningModule subclass *defined* in the
    file. Imported subclasses (`LightningModule` itself, or one from another
    module) don't count — we filter on `obj.__module__`.

    Raises Ft4UserError on any failure, with an actionable message.
    """
    # Lazy: don't pay the Lightning import cost just to parse argv.
    from lightning.pytorch import LightningModule

    # Register under the natural stem (not a prefixed alias) so jsonargparse's
    # importlib.import_module("<stem>") lookup — used by subclass_mode_data
    # to resolve `class_path: <stem>.SomeDataModule` — hits the same module
    # object via sys.modules.
    module_name = model_file.stem
    _guard_against_shadowing(module_name, model_file)
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise Ft4UserError(f"cannot load {model_file} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise Ft4UserError(f"failed to import {model_file}: {e}") from e

    if explicit is not None:
        cls = getattr(module, explicit, None)
        if cls is None:
            raise Ft4UserError(f"class {explicit!r} not found in {model_file}")
        if not (isinstance(cls, type) and issubclass(cls, LightningModule)):
            raise Ft4UserError(f"{explicit!r} is not a LightningModule subclass")
        return cls

    candidates = [
        (name, obj) for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, LightningModule)
        and obj is not LightningModule
        and obj.__module__ == module_name
    ]
    if not candidates:
        raise Ft4UserError(
            f"no LightningModule subclass defined in {model_file}; use "
            f"--model-class if it's imported from elsewhere"
        )
    if len(candidates) > 1:
        names = sorted(n for n, _ in candidates)
        raise Ft4UserError(
            f"multiple LightningModule subclasses defined in {model_file}: "
            f"{names}; use --model-class to disambiguate"
        )
    return candidates[0][1]


def _guard_against_shadowing(module_name: str, model_file: Path) -> None:
    """Refuse to register a model file whose stem collides with an already-
    loaded module (e.g. `torch.py`, `lightning.py`). Otherwise `sys.modules`
    would silently start pointing at the student's file for the rest of the
    interpreter's life, breaking every subsequent `import torch` etc.

    A name that's been registered for THIS same file (re-entry, test
    monkeypatching) is fine — we skip the guard for it.
    """
    if module_name in sys.stdlib_module_names:
        raise Ft4UserError(
            f"model file name {model_file.name!r} collides with the standard "
            f"library module {module_name!r}; rename the file."
        )
    existing = sys.modules.get(module_name)
    if existing is None:
        return
    existing_file = getattr(existing, "__file__", None)
    if existing_file is None or Path(existing_file).resolve() == model_file.resolve():
        return  # same file (re-entry) or a namespace module — safe
    raise Ft4UserError(
        f"model file name {model_file.name!r} collides with the already-"
        f"imported module {module_name!r} (loaded from {existing_file}); "
        f"rename the file."
    )


# ── Cascade args ───────────────────────────────────────────────────────────

def build_cascade_args(
    ft4_args: Ft4Args,
    cascade_paths: list[Path] | None = None,
    *,
    runs_root: Path = Path("runs"),
) -> list[str]:
    """Build the --config layering for LightningCLI. Order (later wins):
      1. src/mlops/ft4.yaml             ft4's operational config
      2. src/mlops/ft4.local.yaml       your overrides to (1), gitignored
      3. src/ft4/models/defaults.yaml     all-model defaults (precision, …)
      4. src/ft4/models/defaults.local.yaml   your overrides to (3), gitignored
      5. <model>.yaml                     per-model config
      6. <model>.local.yaml               your overrides to (5), gitignored
      7. The user's explicit Lightning args from argv

    When `--hset hNNN` is given explicitly (and `--new-hset` is not), the
    saved `<hset>/config.yaml` is substituted for the model.yaml layers
    (5+6). That snapshot is what defined the hset's identity, so this lets
    students resume an old hset even after editing the local model.yaml.

    Missing files are skipped silently. `cascade_paths` is parameterized
    for testability; production code resolves the standard layout via
    `_default_cascade_paths()`.
    """
    if cascade_paths is None:
        cascade_paths = _default_cascade_paths(ft4_args.model_file)
    cascade_paths = list(cascade_paths)
    hset_config = _find_hset_config_path(ft4_args, runs_root)
    if hset_config is not None:
        # Drop the <model>.yaml / <model>.local.yaml layers — the hset
        # snapshot replaces them. Other layers (ft4.yaml, defaults.yaml,
        # …) still apply on top of the snapshot. That's intentional: the
        # snapshot defines this hset's *identity*; operational config can
        # be freshened from the current ft4 install.
        dropped = {
            ft4_args.model_file.with_suffix(".yaml").resolve(),
            ft4_args.model_file.with_suffix(".local.yaml").resolve(),
        }
        cascade_paths = [p for p in cascade_paths if p.resolve() not in dropped]
        cascade_paths.append(hset_config)
        print(
            f"note: --hset {ft4_args.hset}: loading hparams from {hset_config}",
            file=sys.stderr,
        )
    args: list[str] = []
    for path in cascade_paths:
        if path.exists():
            args.extend(["--config", str(path)])
    args.extend(ft4_args.lightning_args)
    return args


def _default_cascade_paths(model_file: Path) -> list[Path]:
    """The standard six cascade paths (before CLI args). Order is later-wins."""
    import ft4.models
    cli_dir = Path(__file__).parent
    models_dir = Path(ft4.models.__file__).parent
    model_yaml = model_file.with_suffix(".yaml")
    model_local = model_file.with_suffix(".local.yaml")
    return [
        cli_dir / "ft4.yaml",
        cli_dir / "ft4.local.yaml",
        models_dir / "defaults.yaml",
        models_dir / "defaults.local.yaml",
        model_yaml,
        model_local,
    ]


def _find_hset_config_path(ft4_args: Ft4Args, runs_root: Path) -> Path | None:
    """If `--hset hNNN` is given (and `--new-hset` is not), look up that
    hset's saved config.yaml under runs/<stem>/<version>/<hset>/. Returns
    None if the hset isn't named, doesn't exist, is ambiguous (multiple
    versions contain it), or its config.yaml is missing — in any of those
    cases the caller proceeds with the normal cascade and the existing
    `resolve_hset` guard handles the error.
    """
    if ft4_args.hset is None or ft4_args.new_hset:
        return None
    stem_dir = runs_root / ft4_args.model_file.stem
    if not stem_dir.is_dir():
        return None
    if ft4_args.version is not None:
        candidates = [stem_dir / ft4_args.version / ft4_args.hset]
    else:
        candidates = [d / ft4_args.hset for d in stem_dir.iterdir() if d.is_dir()]
    matches = [c / "config.yaml" for c in candidates if (c / "config.yaml").is_file()]
    if len(matches) == 1:
        return matches[0]
    return None  # zero matches → resolve_hset errors; multiple → ambiguous


# ── LightningCLI construction ──────────────────────────────────────────────

def build_lightning_cli(ft4_args: Ft4Args, model_class, *, runs_root: Path = Path("runs")):
    """Construct LightningCLI(run=False, save_config_callback=None).

    Path injection and config persistence happen *after* this returns, in
    mlops.run (step 8), via hset_state.inject_paths and persist_config.

    sys.argv is temporarily cleared to suppress Lightning's "args parameter
    is intended to run from within Python" warning when ft4 itself is being
    invoked from a shell (and so sys.argv contains ft4's own flags).
    """
    # Lazy import.
    from lightning.pytorch import LightningDataModule
    from lightning.pytorch.cli import LightningCLI

    cascade = build_cascade_args(ft4_args, runs_root=runs_root)
    _check_cascade_for_todos(cascade)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        return LightningCLI(
            model_class=model_class,
            datamodule_class=LightningDataModule,
            subclass_mode_data=True,
            run=False,
            save_config_callback=None,
            seed_everything_default=random.randint(0, 2**16 - 1),
            args=cascade,
        )
    finally:
        sys.argv = saved_argv


# ── Pre-scan for unfilled !TODO placeholders ───────────────────────────────

# Matches `!TODO` or `!TODO-LAB`, but NOT `!TODOSOMETHING` or `!TODO-LABEL`.
# The negative lookahead enforces that the tag ends here — the next character
# (if any) must not be a YAML-tag-name character (letter, digit, hyphen).
_TODO_TAG_RE = re.compile(r"!TODO(?:-LAB)?(?![A-Za-z0-9-])")


def _check_cascade_for_todos(cascade_args: list[str]) -> None:
    """Bail with a clean Ft4UserError if any cascade yaml contains an
    unfilled `!TODO` / `!TODO-LAB` placeholder.

    Runs after `build_cascade_args` and before LightningCLI is constructed.
    A !TODO in any layered config means a student hasn't completed the lab
    yet; we'd rather list every unfilled placeholder up front than let
    `TodoSentinel` coercion errors trickle out of model construction one
    field at a time.

    No-ops in two situations:
      - The cascade contains `--help` or `-h`: the student is asking what
        flags exist; let jsonargparse render help without forcing them to
        fix every placeholder first (TodoSentinels render fine because no
        coercion happens during --help).
      - The cascade contains no yaml configs at all (test scenarios, or a
        no-cascade invocation).
    """
    if "--help" in cascade_args or "-h" in cascade_args:
        return

    by_file: dict[Path, list[tuple[int, str]]] = {}
    for yaml_path in _config_paths_in_cascade(cascade_args):
        entries = _find_todo_lines(yaml_path)
        if entries:
            by_file[yaml_path] = entries

    if not by_file:
        return

    n = sum(len(v) for v in by_file.values())
    plural = "" if n == 1 else "s"
    msg_lines = [
        f"{n} unfilled lab placeholder{plural} in yaml file(s)\n"
        f"Each is a `!TODO \"...\"` left over from starter code\n"
        f"Complete these before running ft4:",
        "",
    ]
    for path, entries in by_file.items():
        msg_lines.append(f"{path}:")
        lno_width = max(len(str(lno)) for lno, _ in entries)
        for lno, text in entries:
            msg_lines.append(f"{str(lno).rjust(lno_width)}: {text.strip()}")
        msg_lines.append("")
    raise Ft4UserError("\n".join(msg_lines))


def _config_paths_in_cascade(cascade_args: list[str]) -> list[Path]:
    """Extract the yaml paths referenced by `--config` flags in cascade_args.

    Handles both `--config foo.yaml` (space-separated) and `--config=foo.yaml`
    (equals-separated). De-duplicates by resolved path, preserving first-seen
    order so the error message lists files in cascade order.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    i = 0
    while i < len(cascade_args):
        tok = cascade_args[i]
        raw: str | None = None
        if tok == "--config" and i + 1 < len(cascade_args):
            raw = cascade_args[i + 1]
            i += 2
        elif tok.startswith("--config="):
            raw = tok[len("--config="):]
            i += 1
        else:
            i += 1
            continue
        # `Path.resolve()` here also normalises case on case-insensitive
        # filesystems, so the same file referenced two different ways
        # dedupes correctly.
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(Path(raw))  # keep the un-resolved form for nicer error msgs
    return paths


def _find_todo_lines(path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, line_text), …] for every line in `path` carrying a
    `!TODO` or `!TODO-LAB` tag outside a comment. Quietly returns [] if the
    path is unreadable — `build_cascade_args` has already filtered for
    existence, and any other read error will surface clearly when LightningCLI
    tries to consume the file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "!TODO" not in line:  # fast-path filter
            continue
        if line.lstrip().startswith("#"):
            continue
        if _TODO_TAG_RE.search(line):
            out.append((lineno, line))
    return out


if __name__ == "__main__":
    sys.exit(main())
