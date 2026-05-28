# ft4 CLI Design Goals & Spec

*Behavioral goals for the From Tensors to Turing Tests (ft4.ai) MLops command line tool.*

*This document was largely correct when written initally, but **has not been updated**. *When it goes against the code,* **the code is correct.**

*This design doc was used to design the tool, and kept for reference. For tool usage and documentation, see `mlops/README.md`. For implementation details, see `architecture.md`*.

## 1. Overview

`ft4` is the command-line MLops tool used to train and run models in the ft4ai course. It sits on PyTorch Lightning and adds:

- **Smart flow.** A single invocation (`ft4 train my_transformer.py`) detects state, shows the briefing, and trains.
- **Architecture and experiment versioning.** Detects when model architecture, the model file's contents, or behavior hparams change. Organizes outputs so different experiments don't clobber each other.
- **Provenance capture.** Records git state, hparam state, status (completed/interrupted/error), and progress per session.
- **Pedagogy support.** A `!todo` YAML sentinel for unfilled hparams; a documented boolean-hparam pattern for code variants; a `generate_samples` model convention for showing student-visible training progress.

The tool's code lives under `mlops/src/mlops/` and is meant to be readable by students. Installed via `pip install -e .` or `uv sync`.

### Position relative to Lightning

Lightning provides *training extension*: `Trainer.fit(ckpt_path=...)` resumes training from a saved state. Lightning does not provide *experiment versioning* — successive `fit()` calls can have arbitrary hparam differences, and Lightning doesn't track architectural compatibility.

`ft4` adds a thin versioning layer on top: enforces "one hset = one behavior config", autodetects architectural and model-file compatibility, captures provenance per session. Lightning still does the actual training.

---

### Goal

A a run tool for "From Tensors to Turing Tests" (ft4ai) course, which trains the models, stores training output, and generates text (i.e. tells stories). The course code uses PyTorch and Lightning, and so the tool should leverage Lightning CLI.

### Use cases

The tool(s) should be used throughout the lifecycle, which varies. For example:

- Initial "Does this work?" runs.
- Student wants to introduce a new hparam
- Student wants to experiment with new code in model
- Experiments with variants in code, or with different hparams ** Ranging from simple tweaks to disciplined searchs and optimizations ** Ranging from immediate local runs to multiple runs in parallel on cloud GPUs
- Configuring defaults for system (eg based on hardware, eg mixed precision)
- Configuring defaults for a particular model (at different points in lifecycle)
- Running models to generate stories based on prompts

### Desiderata

1. Low Friction. Very important. Initially, a student wants to finish the starter code provided, and run their model. Should be as simple as possible. Ideally, even `$ python src/models/mymodel.py`.
2. Transparency. No magic! Course is to teach, you need to understand your own code.
3. Simplicity. For pedagogical reasons.
4. Encourage, and make easy, iterative development. For example, it should be easy to introduce a new hparam, without having to add boilerplate or declarations.
5. Encourage, and make easy, experimenation, both with hparams and with code variants.
6. Reproducibility and audit trails.
7. Works well with source control. What needs to go into git, goes into git. What needs to stay out, stays out.
8. Supports optimization and searches (though this is secondary). Obviously, we can't attain all of these fully. Still, it's important to know what our goal is. Should make use of Weights & Biases if an account available, but not require it.

## 2. Glossary

Four nested concepts:

- **Model.** A model file (e.g., `src/ft4/models/my_transformer.py`). One file = one model.
- **Version.** A group of hsets that share the same `(state_dict_hash, model_file_hash)`. Tool-detected, autonumbered (`v001`, `v002`, …). Created when either signal changes.
- **Hset.** A behavior-hparam group within a version. All sessions of a hset share the same `config_hash`. Autonumbered (`h001`, `h002`, …).
- **Runconfig**. Includes hyperparameters, like learning rate, that can be changed without invalidating a checkpoint.
- **Session.** One `Trainer.fit()` invocation within a hset. Sessions accumulate. Each session writes its own `<session>/metrics.csv`; aggregated into `<hset>/metrics.csv` at session end.

A **checkpoint** is a `.ckpt` snapshot. Lightning's `ModelCheckpoint` manages `last.ckpt` (always current) plus per-step archives.

**Samples** are generated text written during training for student review. Live at `<hset>/samples/step_NNNNNN.md`. Step numbers are global (`trainer.global_step`), so they correctly sequence across resumed sessions.

---

## 3. Directory Layout

```
project/
├── mlops/                                # tool code (in git)
│   ├── __init__.py
│   ├── main.py                           # entry point, argparse, dispatch
│   ├── run.py                            # smart flow orchestration
│   ├── hset_state.py                    # signals, version/hset/checkpoint resolution
│   ├── sample_callback.py                # SampleGenerationCallback + write_sample_file
│   ├── session_finalization.py           # SessionFinalizationCallback (Lightning hooks)
│   ├── cumulative_metrics.py             # rebuild_cumulative_metrics function
│   ├── briefing_display.py                  # render_briefing (plain text → stderr)
│   ├── briefing_display.py               # render_briefing / render_briefing_plain
│   ├── todo_yaml.py                      # !todo PyYAML constructor
│   └── ft4.yaml                     # system defaults (config cascade base)
│
├── src/ft4/models/                       # course models (in git)
│   ├── iris.py + iris.yaml
│   ├── transformer.py + transformer.yaml
│   ├── attention.py                      # reused by transformer
│   └── mlp.py                            # reused by iris + transformer
│
├── runs/                                 # gitignored
│   └── transformer/                      # one subdir per model
│       └── v001/
│           ├── version.json              # state_dict_hash, model_file_hash, vdesc, git
│           ├── model.py.snapshot         # model file content at version creation
│           └── h001/
│               ├── hset.json            # config_hash, hdesc, last_session_at
│               ├── config.yaml           # Lightning's resolved config
│               ├── sessions.jsonl        # append-only session log
│               ├── metrics.csv           # cumulative across sessions; integer `session` column
│               ├── checkpoints/
│               │   ├── last.ckpt
│               │   └── step={N}.ckpt
│               ├── sessions/
│               │   └── sNNN/
│               │       ├── metrics.csv   # per-session metrics
│               │       └── hparams.yaml  # Lightning's artifact
│               └── samples/              # only if model defines generate_samples
│                   ├── step_000000.md    # phase-1 baseline (fresh hsets only)
│                   └── step_NNNNNN.md    # one per validation epoch
└── pyproject.toml                        # [project.scripts] ft4 = "mlops.main:main"
```

The tool never deletes anything in `runs/`. Disk cleanup is the user's responsibility.

---

## 4. CLI Surface

Single entry point `ft4` declared in `pyproject.toml`:

```toml
[project.scripts]
ft4 = "mlops.main:main"

[tool.setuptools.package-data]
"mlops" = ["*.yaml"]
```

### 4.1 Invocation forms


| Form                                      | Used when                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `uv run ft4 train my_transformer.py`      | uv user                                                                          |
| `ft4 train my_transformer.py`             | venv activated; entry point on PATH                                              |
| `python src/ft4/models/my_transformer.py` | Runs the model file directly; prints a redirect message and exits 2 (see § 4.4) |

### 4.2 Subcommands

V1 has one functional subcommand: `run`. The other subcommands are parsed (so `--help` works) but raise `NotImplementedError`:


| Command                   | Status                                                                                                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ft4 train <model.py>`    | Implemented. Phase 0 (read-only resolve) → fit (with phase-1 baseline samples and per-validation samples if model supports it). Briefing prints at on_train_start. |
| `ft4 train <model.py>`    | Deferred. Will be`run` minus phase-1 baseline.                                                                                                                      |
| `ft4 generate <model.py>` | Deferred. Load checkpoint, call`generate_samples()`, print.                                                                                                         |
| `ft4 validate <model.py>` | Deferred. Load checkpoint,`trainer.validate()`.                                                                                                                     |
| `ft4 metrics <model.py>`  | Deferred. Summarize`metrics.csv`.                                                                                                                                   |
| `ft4 list <model.py>`     | Deferred. Show all versions + hsets with descriptions.                                                                                                              |

There is no `ft4` invocation without an explicit subcommand. (The original spec floated this; deferred.)

### 4.3 Flags


| Flag                                         | Effect                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `--hset hNNN`                                | Pin to existing hset. For training: config_hash mismatch errors.                           |
| `--version vNNN`                             | Pin to existing version (must be state_dict-compatible).                                   |
| `--ckpt {last                                | <path>}`                                                                                   |
| `--new-version`                              | Force a new version. Use when an edit outside the model file affects behavior.             |
| `--vdesc "..."`                              | Set/update version description (stored in`version.json`).                                  |
| `--hdesc "..."`                              | Set/update hset description (stored in`hset.json`).                                        |
| `--model-class <name>`                       | Disambiguate when a file defines >1`LightningModule` subclass.                             |
| `--prompt "..."`                             | Override generation prompt(s) for this invocation. Repeatable.                             |
| `--wandb`                                    | Enable wandb experiment tracking. Errors if wandb isn't installed or you're not logged in. |
| `--model.X=Y`, `--trainer.X=Y`, `--data.X=Y` | jsonargparse-syntax overrides into the resolved config.                                    |

`--strict` from the original spec is dropped. The default flow is non-interactive — no confirm prompt — and the rich-prompt machinery the spec described isn't needed.

### 4.4 The in-file `__main__` block

Course models include a 3-line `__main__` block that redirects students to the proper tool:

```python
if __name__ == "__main__":
    import sys
    print(
        "This file is an ft4 model. Run it through ft4:\n"
        f"\n    ft4 train {sys.argv[0]}\n"
        "\n(See https://ft4.ai for the course; `ft4 --help` for command-line options.)",
        file=sys.stderr,
    )
    sys.exit(2)
```

The block doesn't auto-run anything. It's deliberately a sign saying "use the door," not a transparent shortcut. (Option c, not d.) The pedagogy rationale: a student typing `python iris.py` learns the tool's name immediately rather than silently bypassing the framework.

### 4.5 Model class discovery

1. Import the file at the given path under its natural module stem.
2. Find all `LightningModule` subclasses whose `__module__` matches the imported module name (excludes imports from elsewhere).
3. Zero matches → error: *"No LightningModule subclass defined in {file}."*
4. One match → use it.
5. More than one → error unless `--model-class <name>` is given.

---

## 5. Hparam Cascade

Three tiers, later overriding earlier:

1. **`mlops/src/mlops/defaults.yaml`** (system). Trainer settings, default logger, default callbacks.
2. **`<model>.yaml`** (sibling of model file). Architecture and training hparams.
3. **`--model.X=Y`** etc. (CLI). One-off overrides.

`<model>.yaml` is discovered as `<model_path>.yaml` (e.g., `transformer.py` → `transformer.yaml`).

The fully-resolved config is saved to `<hset>/config.yaml` at session start.

### 5.1 `defaults.yaml`

```yaml
trainer:
  precision: bf16-mixed
  enable_progress_bar: true
  enable_model_summary: true
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: "runs"        # overwritten per-session by inject_paths
      name: ""
      version: ""
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null         # overwritten per-hset by inject_paths
        filename: "{step}"
        save_last: true
        save_top_k: -1
        monitor: null
    - class_path: mlops.sample_callback.SampleGenerationCallback
      init_args:
        epoch_interval: 1     # samples every validation epoch
        step_interval: 0      # disabled
    - class_path: mlops.session_finalization.SessionFinalizationCallback
```

The `dirpath`/`save_dir` placeholders are mutated at runtime by `inject_paths` after the LightningCLI builds the trainer. `SessionFinalizationCallback` is required; if a `<model>.yaml` overrides the callbacks list, it must include this callback to retain session-record and hset-aggregation behavior.

### 5.2 `!todo` YAML sentinels

For places students must fill in:

```yaml
model:
  dim: !todo "unit2.lab1 — choose an embedding dimension"
  lr: 3e-4
```

`!todo` returns a sentinel whose `__int__`, `__float__`, `__bool__`, `__index__`, and `__str__` raise `NotImplementedError` carrying the tag's string. Author writes against a real value via the existing preprocessor; `ft4` knows nothing about the preprocessor and only registers the `!todo` PyYAML constructor.

---

## 6. The Smart Flow

`ft4 train my_transformer.py` does, in order:

1. **Phase 0 (read-only setup).** Discover the model class. Build LightningCLI (instantiates model + datamodule from the config cascade). Compute `(state_dict_hash, model_file_hash, config_hash)`. Resolve version → hset → checkpoint. Read existing sessions.jsonl. Build the briefing (Rich Panel + plain-text fallback) that `BriefingPrintCallback` will print at `on_train_start`.
2. **Phase 2 (destructive).**

   - Persist resolved config to `<hset>/config.yaml`.
   - Inject prompts onto callbacks that want them (duck-typed on `prompts` attribute).
   - Phase-1 baseline: if `ckpt_path is None` (fresh hset) AND model defines `generate_samples`, write `<hset>/samples/step_000000.md` *before* creating the session directory.
   - Create `<hset>/sessions/sNNN/`.
   - Mutate trainer's `ModelCheckpoint.dirpath` → `<hset>/checkpoints/`, logger's `save_dir` → `<session>/`.
   - Start `SessionFinalizationCallback` with per-run state (started_at, git_commit, git_dirty).
   - Call `trainer.fit(..., ckpt_path=...)`. Lightning handles training; on clean completion `on_fit_end` fires; on Ctrl-C or error, `on_exception` then `teardown` fire. Either path lands in `SessionFinalizationCallback._finalize`, which appends to `sessions.jsonl` (with `status` field) and rebuilds `<hset>/metrics.csv`.

### 6.1 Version resolution

**Write ops:**

1. If `--new-version`, autonumber unconditionally.
2. Else scan `runs/<model>/` for highest `vNNN`.
3. If its `version.json` matches current `(state_dict_hash, model_file_hash)`, reuse.
4. Else autonumber `v{N+1:03d}`, write `version.json` and `model.py.snapshot`.

**Read ops** (deferred subcommands): scan newest-first by mtime, find compatible match.

### 6.2 Hset resolution

**If `--hset hNNN`:**

- Doesn't exist → error.
- For training: `config_hash` mismatch → error.

**If `--hset` unset:**

- Scan hsets in current version newest-first by mtime.
- First hset whose `config_hash` matches → reuse.
- No match → autonumber `t{N+1:03d}`.

### 6.3 Checkpoint resolution

`--ckpt last` (default): `<hset>/checkpoints/last.ckpt`. Missing → train from scratch.

`--ckpt <path>`: explicit; loaded verbatim. Useful for cross-hset fine-tuning (point at another hset's checkpoint to seed this one).

`--ckpt best`: deferred to V2.

### 6.4 Stats display

Plain text to stderr, no rich. Example:

```
ft4: transformer.py / v002 / h001
  version: 8b1e7a9f
  hset:   3 sessions, last 12 minutes ago
  ckpt:    last.ckpt (step 9423)
```

`render_briefing` reads `sessions.jsonl` for the session count and last-session timestamp.

---

## 7. Detection (Three Signals)

### 7.1 `state_dict_hash`

SHA-256 of sorted `(parameter_name, shape, dtype)` tuples (including buffers) from a freshly-instantiated model. Detects architecture changes.

Mismatch on write → new version. On read → search older versions for match (deferred).

### 7.2 `model_file_hash`

SHA-256 of the model file's contents. Detects any change — committed or not, behavioral or cosmetic.

Mismatch on write → new version (model.py.snapshot updated). On read → warning; proceed with the existing version's checkpoint.

**Caught:** any edit to the model file. **Missed:** edits to files the model imports (e.g., sibling `attention.py`). Mitigation: `--new-version`.

### 7.3 `config_hash`

Hash of the resolved config dict with `CONFIG_HASH_EXCLUDED_PREFIXES` removed via prefix match. The exclusion list (in `hset_state.py`):

```python
CONFIG_HASH_EXCLUDED_PREFIXES = frozenset({
    # Stopping criteria — when to stop training, not what training is.
    "trainer.max_epochs",
    "trainer.max_steps",
    "trainer.max_time",
    "trainer.val_check_interval",
    "trainer.check_val_every_n_epoch",
    "trainer.log_every_n_steps",
    # Infrastructure — observers, not actors.
    "trainer.callbacks",
    "trainer.logger",
    # UI — doesn't affect training results.
    "trainer.enable_progress_bar",
    "trainer.enable_model_summary",
    # Path-bound — ft4 injects these per-session.
    "trainer.default_root_dir",
    # Debugging / dev modes — not training behavior.
    "trainer.fast_dev_run",
    "trainer.detect_anomaly",
    "trainer.profiler",
    "trainer.barebones",
    # Performance autotuning — not scientific.
    "trainer.benchmark",
})
```

Prefix matching means `trainer.logger` filters `trainer.logger.class_path`, `trainer.logger.init_args.save_dir`, etc. The scientifically meaningful trainer keys (`precision`, `gradient_clip_val`, `accumulate_grad_batches`, `deterministic`, `seed_everything`) remain in the hash.

Trade-off accepted: a user who adds `EarlyStopping` as a callback in their `<model>.yaml` won't fork a hset. If a callback change should count as a scientific change, the user signals it via a model hparam or `--vdesc`.

`config_hash` mismatch → new hset. The basis for hset resolution (§ 6.2).

---

## 8. TTY Policy

The default flow is non-interactive — no confirm prompt, no `select` on stdin, no platform-specific input. TTY-awareness only affects rendering:


| Stream | TTY                     | Behavior                                                                    |
| -------- | ------------------------- | ----------------------------------------------------------------------------- |
| stdout | TTY                     | Briefing + sample panels render as Rich Panels above the live progress bar. |
| stdout | non-TTY (`> log.txt`, ` | tee`, etc.)                                                                 |

---

## 9. Pedagogy Patterns

### 9.1 `!todo` (covered in § 5.2)

### 9.2 Boolean hparams for code variants

Two toggles in the reference Transformer, both Flavor 1 (forward-pass only; no parameter shape change):

```python
class Transformer(L.LightningModule):
    def __init__(self, ..., use_pos_enc=False, use_logit_softcap=False, ...):
        super().__init__()
        self.save_hyperparameters()  # required: lands toggles in config_hash
        # ...

    def _token2vec(self, tokens):
        x = self.embed(tokens)
        if self.hparams.use_pos_enc:
            x = x + self.positional_encoding[:x.shape[1]]
        return x

    def _y2logits(self, y):
        logits = self.unembed(y)
        if self.hparams.use_logit_softcap:
            logits = LOGIT_SOFTCAP * (logits / LOGIT_SOFTCAP).tanh()
        return logits
```

Flipping either toggle changes `config_hash` only — same `state_dict_hash` → same version, new hset. Students see "use_pos_enc=true vs false" as side-by-side hsets.

(Flavor 2 — parametric toggles that fork versions — is supported by the framework but no example ships in V1. Reference: spec rev 6, dropped per "best practice" pedagogy review.)

### 9.3 `generate_samples` convention

A LightningModule may define:

```python
def generate_samples(self, prompts: list[str] | None = None) -> str:
    """Return markdown content for display to the student."""
```

If present, ft4 writes:

- One sample at `step_000000.md` before training begins, for fresh hsets only.
- One sample at `step_NNNNNN.md` after each validation epoch (skipping Lightning's sanity-check validation).
- All samples written to `<hset>/samples/`.

The convention is opt-in: classifiers and other non-generative models simply don't define the method. The `SampleGenerationCallback` silently no-ops in that case.

`prompts` come from `--prompt "..."` on the CLI (repeatable). If unset, the model's default prompts are used.

For the reference Transformer, `generate_samples` uses `LangGen` (the project's existing autoregressive decoder):

```python
def generate_samples(self, prompts=None, max_to_generate=60):
    from ft4.lang_gen import LangGen
    prompts = prompts or DEFAULT_SAMPLE_PROMPTS
    was_training = self.training
    self.eval()
    try:
        # ... iterate LangGen over each prompt, return formatted markdown
    finally:
        if was_training:
            self.train()
```

---

## 10. Logger Compatibility

V1: CSV only. The defaults.yaml ships `lightning.pytorch.loggers.CSVLogger` pointing at the session directory.

Each session writes its own `<session>/metrics.csv`. After fit ends (clean, interrupted, or errored), `SessionFinalizationCallback` calls `rebuild_cumulative_metrics(hset_dir)`, which concatenates all per-session files into `<hset>/metrics.csv` with a leading integer `session` column.

TensorBoard and WandB are deferred. The architecture is logger-agnostic; future loggers will write to `<session>/` directories that the user's tool (TB UI, WandB UI) can read directly. There is no plan to aggregate non-CSV logger output at the hset level — those tools have their own multi-run UIs.

---

## 11. Schemas

### `version.json`

```json
{
  "created_at": "2026-05-14T14:30:00Z",
  "description": "Initial attempt",
  "state_dict_hash": "def456…",
  "model_file_hash": "ghi789…",
  "git_commit_at_creation": "a3f9d21",
  "git_dirty_at_creation": false
}
```

### `hset.json`

```json
{
  "created_at": "2026-05-14T14:32:00Z",
  "last_session_at": "2026-05-14T16:45:00Z",
  "description": "Trying smaller dim",
  "config_hash": "abc123…"
}
```

`last_session_at` is updated by `SessionFinalizationCallback` after each session.

### `sessions.jsonl`

Append-only, one JSON object per line, written at session end:

```json
{
  "session_name": "s001",
  "session_index": 1,
  "started_at": "2026-05-14T14:32:00Z",
  "ended_at": "2026-05-14T15:30:00Z",
  "epochs": 2,
  "steps": 6282,
  "final_train_loss": 0.51,
  "final_val_loss": 0.84,
  "all_metrics": {"train_ce": 0.51, "val_ce": 0.84},
  "git_commit": "a3f9d21",
  "git_dirty": false,
  "status": "completed"
}
```

Fields:

- `status` is one of `"completed"`, `"interrupted"`, `"error"`.
- If `status == "error"`, an additional `error` field carries `"TypeName: message"`.
- `git_patch_hash` was in spec rev 6; dropped (use git itself for diffs if needed).
- Spec rev 6's `target_max_*` / `achieved_*` are dropped; `epochs` and `steps` are sufficient.

### `<hset>/metrics.csv`

Cumulative metrics across all sessions, with a leading integer `session` column. Schema evolution between sessions is handled by outer-joining columns (a row from a session that didn't log column X gets an empty cell for X).

Atomic write: `metrics.csv.tmp` → rename. A crashed run never leaves a partially-written file.

---

## 12. Cross-Platform Notes

- **Linux, macOS, WSL**: primary. Everything works.
- **Windows native**: `pathlib.Path` everywhere; no symlinks; rich-style prompts not used.

No symlinks. No `select` on stdin.

---

## 13. Limits and Accepted Compromises

**Detection envelope.**

- Architecture (Signal 1): conclusive.
- Model file (Signal 2): conclusive for the file itself. Edits to imported files not detected. `--new-version` is the override.
- Config (Signal 3): hash with `CONFIG_HASH_EXCLUDED_PREFIXES` removed.

**Cosmetic edits fork versions.** A comment change to the model file produces a new `model_file_hash` and thus a new version. Cheap in disk, conceptually honest. The user can manually copy a checkpoint via `--ckpt /old/path/last.ckpt` if they want to continue training the old weights in the new version.

**Schedules tied to stopping criteria are subtle.** If a model uses an LR schedule that depends on `max_epochs` (e.g., cosine annealing), bumping `max_epochs` between sessions changes the LR trajectory — and this is *not* a `config_hash` change (stopping criteria are excluded). The stats display does not currently flag this; documenting as a known footgun.

**Opaque hset and version names.** `h001`, `v001` don't describe contents. `--hdesc`/`--vdesc` provide human-readable descriptions; surfaced in `ft4 list` (deferred subcommand).

**Models must define architecture in `__init__`.** Lazy `setup()`-time module construction won't be reflected in `state_dict_hash` at instantiation time. Rare in coursework.

**No parallelism.** Multiple concurrent `ft4` invocations on the same hset directory will race and corrupt files.

**Phase-1 baseline only fires for fresh hsets.** Resumed sessions don't get a `step_000000.md` rewrite (the model state at session start equals the prior session's end state, already captured at that step number).

---

## 14. Reference Models

`src/ft4/models/iris.py` — hello-world classifier. No `generate_samples`. Tests in `mlops/tests/test_iris.py` confirm threshold accuracy.

`src/ft4/models/transformer.py` — reference language model. Two Flavor-1 toggles. `generate_samples` via `LangGen`. Default config in `transformer.yaml` (vocab_size, dim, depth, expansion_factor, lr, weight_decay, warmup_steps, both toggles).

Both files end with the option-c `__main__` block (§ 4.4).
