"""
# Test everything
pytest

# Test only current lab
pytest --only-lab

# Test only Unit 1 Lab 2
pytest --only-lab unit1.lab2

# Test everything up to and including Unit 1 Lab 2
pytest --upto-lab unit1.lab2
"""

import os
import re
import subprocess
import warnings
from typing import Iterable, Optional, Tuple, List, Set

import pytest
from _pytest.warning_types import PytestWarning

from test_helpers import find_checkpoint

# Tests must never reach wandb. Set the mode *before* any wandb import so
# every subprocess we spawn inherits it. wandb is opt-in via --wandb, so
# tests that don't pass that flag never reach the network anyway — this is
# the belt-and-suspenders for any test that exercises the wandb path.
os.environ.setdefault("WANDB_MODE", "disabled")

_LAB_RE = re.compile(r"^unit(\d+)\.lab(\d+)$")
_AUTO = "__AUTO__"  # sentinel when flag provided with no value


# ----------------------------- CLI options -----------------------------

def pytest_addoption(parser):
    # Optional values:
    #   - flag absent -> option is None
    #   - flag present with no value -> option is _AUTO
    #   - flag present with value -> option is that string
    parser.addoption(
        "--only-lab",
        action="store",
        nargs="?",
        const=_AUTO,
        default=None,
        help=("Run only tests belonging to the specified lab (or current lab if none specified). "
              "Examples: '--only-lab' (current lab), '--only-lab unit1.lab1', '--only-lab unit1.lab1,unit1.lab3'")
    )
    parser.addoption(
        "--upto-lab",
        action="store",
        nargs="?",
        const=_AUTO,
        default=None,
        help=("Run all tests up to and including the specified lab (or current lab if none specified). "
              "Examples: '--upto-lab' (current lab), '--upto-lab unit1.lab1'")
    )
    parser.addoption(
        "--skip-checkpoint-tests",
        action="store_true",
        default=False,
        help=("Skip tests that require a trained checkpoint instead of failing them. "
              "Also enabled via FT4_SKIP_CHECKPOINT_TESTS=1.")
    )


# ----------------------------- Utilities ------------------------------

def _parse_lab_id(lab_id: str) -> Optional[Tuple[int, int]]:
    """Parse 'unitX.labY' -> (X, Y) as ints; return None if invalid."""
    m = _LAB_RE.match(lab_id or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _split_ids(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _format_lab_tuple(tup: Tuple[int, int]) -> str:
    return f"unit{tup[0]}.lab{tup[1]}"


def _latest_valid(ids: Iterable[str]) -> Optional[Tuple[int, int]]:
    """Return the max (unit, lab) among valid ids; warn on invalid; None if none valid."""
    parsed: List[Tuple[int, int]] = []
    invalid: List[str] = []
    for lab_id in ids:
        t = _parse_lab_id(lab_id)
        if t is None:
            invalid.append(lab_id)
        else:
            parsed.append(t)
    if invalid:
        warnings.warn(
            PytestWarning(
                "Ignoring invalid lab id(s): "
                + ", ".join(repr(x) for x in invalid)
                + " (expected 'unitN.labM')"
            )
        )
    return max(parsed) if parsed else None


def _detect_current_lab() -> Optional[Tuple[int, int]]:
    """
    Call './whichlab -c':
      - On success, stdout is 'unitX.labY\\n' -> return (X, Y).
      - If 'None' or non-zero exit, return None.
    """
    try:
        out = subprocess.check_output(["./whichlab", "-c"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if out == "None" or not out:
        return None
    t = _parse_lab_id(out)
    return t


def _write_line(config, msg: str) -> None:
    """Write a single line to terminal if reporter exists; otherwise print()."""
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line(msg)
    else:
        print(msg)


# -------------------------- Early option resolve -----------------------

def pytest_configure(config):
    only_arg = config.getoption("--only-lab")   # None | _AUTO | "ids"
    upto_arg = config.getoption("--upto-lab")   # None | _AUTO | "ids"

    # Checkpoint-test skip banner + warning. Done first so it always fires, even when no
    # --only-lab / --upto-lab flag is present (which short-circuits below).
    if _checkpoint_tests_skipped(config):
        msg = ("Checkpoint tests (of trained models) were be skipped. "
               "Train model with `ft4 train` and remove --skip-checkpoint-tests to run them.")
        _write_line(config, msg)
        # issue_config_time_warning routes into pytest's end-of-session warnings summary
        # (rather than Python's default warnings handler that prints immediately).
        config.issue_config_time_warning(PytestWarning(msg), stacklevel=2)

    # Mutually exclusive
    if (only_arg is not None) and (upto_arg is not None):
        raise pytest.UsageError("Use either --only-lab or --upto-lab, not both.")

    # Defaults (no flags): run all tests untouched; nothing to compute/store.
    if only_arg is None and upto_arg is None:
        config._lab_only_ids = None           # type: ignore[attr-defined]
        config._lab_upto_latest = None        # type: ignore[attr-defined]
        return

    # Resolve --only-lab
    only_ids: Set[str] | None = None
    if only_arg is _AUTO:
        current = _detect_current_lab()
        if current is None:
            pytest.exit(
                "Error: No lab started.\n"
                "To start the first lab, run:\n"
                "  ./startnextlab",
                returncode=2,
            )
        only_ids = {_format_lab_tuple(current)}
    elif isinstance(only_arg, str):
        candidates = set(_split_ids(only_arg))
        valid = {cid for cid in candidates if _parse_lab_id(cid) is not None}
        if candidates and not valid:
            pytest.exit(
                "Error: All --only-lab ids were invalid (expected 'unitN.labM').",
                returncode=2,
            )
        if candidates and len(valid) < len(candidates):
            invalid = sorted(candidates - valid)
            warnings.warn(
                PytestWarning(
                    "Ignoring invalid --only-lab id(s): " + ", ".join(repr(x) for x in invalid)
                )
            )
        only_ids = valid if valid else None
    else:
        only_ids = None

    # Resolve --upto-lab
    upto_latest: Optional[Tuple[int, int]] = None
    if upto_arg is _AUTO:
        current = _detect_current_lab()
        if current is None:
            pytest.exit(
                "Error: No lab started.\n"
                "To start the first lab, run:\n"
                "  ./startnextlab",
                returncode=2,
            )
        upto_latest = current
    elif isinstance(upto_arg, str):
        upto_latest = _latest_valid(_split_ids(upto_arg))
        if upto_arg and upto_latest is None:
            warnings.warn(
                PytestWarning(
                    "All --upto-lab ids were invalid; proceeding without lab gating."
                )
            )
    else:
        upto_latest = None

    # Store resolved selections for later hook
    config._lab_only_ids = only_ids           # type: ignore[attr-defined]
    config._lab_upto_latest = upto_latest     # type: ignore[attr-defined]

    # Friendly summary
    if only_ids:
        labs_list = ", ".join(sorted(only_ids))
        _write_line(config, f"Running tests marked {labs_list} only")
    elif upto_latest is not None:
        _write_line(config, f"Running tests up to {_format_lab_tuple(upto_latest)} (later labs deselected)")


def _checkpoint_tests_skipped(config) -> bool:
    """True if checkpoint tests should skip (rather than fail) when no checkpoint exists."""
    return bool(config.getoption("--skip-checkpoint-tests")) or \
           os.getenv("FT4_SKIP_CHECKPOINT_TESTS") == "1"


# --------------------------- Collection filtering ----------------------

def pytest_collection_modifyitems(config, items):
    only_ids: Optional[Set[str]] = getattr(config, "_lab_only_ids", None)
    upto_latest: Optional[Tuple[int, int]] = getattr(config, "_lab_upto_latest", None)

    # No selection flags -> no lab filtering, but still run the missing-checkpoint banner below.
    if only_ids is None and upto_latest is None:
        _maybe_emit_missing_checkpoint_banner(config, items)
        return

    deselected: List[pytest.Item] = []
    kept: List[pytest.Item] = []

    for item in items:
        m = item.get_closest_marker("lab")
        if m is None or not m.args:
            # Unmarked tests
            if only_ids is not None:
                deselected.append(item)  # --only-lab excludes unmarked
            else:
                kept.append(item)        # --upto-lab includes unmarked
            continue

        lab_id = str(m.args[0])
        lab_tuple = _parse_lab_id(lab_id)
        if lab_tuple is None:
            # Bad marker -> warn and treat as unmarked
            warnings.warn(
                PytestWarning(
                    f"Ignoring invalid lab marker {lab_id!r} on {item.nodeid} "
                    f"(expected 'unitN.labM')"
                )
            )
            if only_ids is not None:
                deselected.append(item)
            else:
                kept.append(item)
            continue

        # --only-lab: keep exact matches only
        if only_ids is not None:
            if lab_id in only_ids:
                kept.append(item)
            else:
                deselected.append(item)
            continue

        # --upto-lab: deselect future labs
        if upto_latest is not None:
            if lab_tuple <= upto_latest:
                kept.append(item)
            else:
                deselected.append(item)
            continue

        # Fallback (shouldn't hit)
        kept.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    # Detect missing checkpoints once, up front, and emit a single grouped banner.
    # Per-test failures stay terse ("No checkpoint for 'X'") so the short summary is readable
    # even with many parametrized cases.
    _maybe_emit_missing_checkpoint_banner(config, items)


def _maybe_emit_missing_checkpoint_banner(config, items) -> None:
    if _checkpoint_tests_skipped(config):
        return  # the skip-tests warning already covers this case
    missing: Set[str] = set()
    for item in items:
        marker = item.get_closest_marker("needs_checkpoint")
        if marker is None or not marker.args:
            continue
        model_stem = str(marker.args[0])
        if find_checkpoint(model_stem) is None:
            missing.add(model_stem)
    if not missing:
        return
    stems = sorted(missing)
    train_lines = "\n".join(f"  ft4 train src/ft4/models/{s}.py" for s in stems)
    banner = (
        f"Missing checkpoints for: {', '.join(stems)}. Train with:\n"
        f"{train_lines}\n"
        f"Or rerun pytest with --skip-checkpoint-tests to skip these tests."
    )
    _write_line(config, banner)


# ---------------------------- Checkpoint fixture ----------------------------

@pytest.fixture
def require_checkpoint(request):
    """Resolve a checkpoint path or fail (or skip, with the opt-out flag).

    The model_stem is read from the test's `@pytest.mark.needs_checkpoint('<stem>')`
    marker if not passed explicitly. The marker form is preferred because it lets us
    detect all missing checkpoints once at collection time and emit a single grouped banner.

    Usage:
        @pytest.mark.needs_checkpoint("neural_ngram")
        def test_something(require_checkpoint):
            ckpt = require_checkpoint()
            model = load_ckpt_cpu(ckpt, NeuralNgram)
    """
    def _get(model_stem: str | None = None) -> str:
        if model_stem is None:
            marker = request.node.get_closest_marker("needs_checkpoint")
            if marker and marker.args:
                model_stem = str(marker.args[0])
        if model_stem is None:
            raise pytest.UsageError(
                "require_checkpoint() called without a model_stem and the test has no "
                "@pytest.mark.needs_checkpoint('<model_stem>') marker."
            )
        ckpt = find_checkpoint(model_stem)
        if ckpt is None:
            # Short, single-line message: the detailed `ft4 train ...` instructions live in
            # the collection-time banner so they aren't repeated per parametrized case.
            short = f"No checkpoint for {model_stem!r}: train via `ft4 train` or `--skip-checkpoint-tests`"
            if _checkpoint_tests_skipped(request.config):
                pytest.skip(short)
            else:
                pytest.fail(short)
        return ckpt
    return _get
