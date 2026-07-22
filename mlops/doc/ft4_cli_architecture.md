# ft4 CLI Architecture

This document orients a new developer (or AI) to the ft4 CLI codebase. It explains *how* the code is laid out and *why* — not *what* ft4 does from a user's perspective (see `spec.md` for that). Read this when you're about to make a change and want to know where it goes.

**Note:** This document was largely correct when written, but has not been updated. When it goes against the code, **this doc is wrong**, and **the code is correct**.

---

## 1. Code Map

All tool code lives under `mlops/src/mlops/`. Each module is a single concern.

| Module | Lines | What it owns |
|--------|-------|--------------|
| `main.py` | ~300 | Entry point, argparse, jsonargparse split, LightningCLI construction, model-class discovery, defaults.yaml resolution, subcommand dispatch. |
| `run.py` | ~150 | The `run` subcommand's orchestration (Phase 0 → Phase 2 fit). Owns the `_build_session_record` helper used by `session_finalization.py`. |
| `hset_state.py` | ~530 | Signals (state_dict, model_file, config hashes), version resolution, hset resolution, checkpoint resolution, sessions.jsonl I/O, path injection. Pure logic; no Lightning callbacks. |
| `sample_callback.py` | ~110 | `SampleGenerationCallback` Lightning callback + `write_sample_file` helper. |
| `live_sample_callback.py` | ~150 | `LiveSampleCallback`: prints sample panel (TTY) or plain text (non-TTY) after each validation. |
| `session_finalization.py` | ~110 | `SessionFinalizationCallback` Lightning callback. Bridges Lightning's on_exception/on_fit_end/teardown to session-record + cumulative-metrics writes. |
| `cumulative_metrics.py` | ~70 | `rebuild_cumulative_metrics` function. Concatenates session metrics into hset-level CSV. No callback. |
| `briefing_display.py` | ~250 | `render_briefing(...)` (Rich Panel) and `render_briefing_plain(...)` (plain text) for the hset-state briefing. |
| `briefing_print_callback.py` | ~50 | `BriefingPrintCallback`: prints the briefing once at on_train_start (panel or plain). |
| `todo_yaml.py` | ~40 | `TodoSentinel` class + PyYAML SafeLoader constructor registration. |
| `defaults.yaml` | ~40 | System-level config defaults. |

Total: ~1500 lines tool code, ~3000 lines tests.

---

## 2. The Two-Layer Boundary

The codebase has a hard internal boundary: **Lightning-aware** vs **Lightning-free**.

| Lightning-free (Layer 1) | Lightning-aware (Layer 2) |
|--------------------------|---------------------------|
| `hset_state.py` (the path-injection seam is the only Lightning touch — duck-typed, no import) | `sample_callback.py` |
| `todo_yaml.py` | `live_sample_callback.py` |
| `briefing_display.py` | `briefing_print_callback.py` |
| `main.py`'s argparse half | `session_finalization.py` |
|  | `cumulative_metrics.py` (no, see below) |
|  | `main.py`'s `build_lightning_cli` |
|  | `run.py` |

Why the split: Lightning is the largest external dependency. Keeping the core logic (signals, versioning, hset state, JSON I/O) Lightning-free makes it unit-testable in isolation and means a future "ft4-core" extraction (mentioned in the original spec § 15) would be a clean cut.

`cumulative_metrics.py` is in Layer 1 — no Lightning import — by design. It's a function over filesystem paths.

---

## 3. Lifecycle (the `run` flow)

```
ft4 train model.py [--flags]
    │
    ▼
main.main(argv)
    │
    ├── todo_yaml.register_constructor()  # !todo PyYAML hook
    ├── split_argv(argv)  ─→  ft4_argv, lightning_argv
    ├── _build_parser().parse_args(ft4_argv)  ─→  Ft4Args
    └── _dispatch(args, lightning_argv)  →  run.run(args)
                                            │
                                            ▼
                                    ┌── PHASE 0 (read-only) ────────────────────┐
                                    │   discover_model_class                    │
                                    │   build_lightning_cli (LightningCLI(...)) │
                                    │       → instantiates model + datamodule  │
                                    │   compute_signals                         │
                                    │   resolve_version → resolve_hset         │
                                    │   resolve_checkpoint                      │
                                    │   read_sessions                           │
                                    │   render_briefing (Panel + plain text)    │
                                    │   _would_be_noop check                    │
                                    └───────────────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌── PHASE 2 (destructive) ──────────────────┐
                                    │   persist_config → <hset>/config.yaml   │
                                    │   inject prompts onto callbacks           │
                                    │   phase-1 baseline samples (if fresh)     │
                                    │   create_session_dir                      │
                                    │   inject_paths (logger, checkpoint, cbs)  │
                                    │   SessionFinalizationCallback.start(...)  │
                                    │   trainer.fit(...)                        │
                                    │       │                                   │
                                    │       ├── (clean) on_fit_end → teardown   │
                                    │       └── (Ctrl-C / err) on_exception     │
                                    │                          → teardown       │
                                    │   ▼                                       │
                                    │   SessionFinalizationCallback._finalize:  │
                                    │     ─ rebuild_cumulative_metrics          │
                                    │     ─ append session record to sessions.jsonl
                                    │     ─ update_hset_last_session           │
                                    └───────────────────────────────────────────┘
```

The Phase 0 / Phase 2 split is a hard line: nothing in Phase 0 mutates the filesystem (except `runs/<model>/v<NNN>/` creation for new versions; deliberate exception). The user can Ctrl-C during Phase 0 with zero artifacts left behind. Anything irreversible happens only in Phase 2.

---

## 4. Key Abstractions

### 4.1 `Ft4Args` (frozen dataclass)

In `hset_state.py`. The single source of truth for what the user asked for. Fields: `subcommand`, `model_file`, `model_class`, `hset`, `version`, `ckpt`, `new_version`, `vdesc`, `hdesc`, `prompt`, `compile`, `wandb`, `lightning_args`. Constructed in `main.py` from argparse; threaded through `run.py`.

### 4.2 `Signals` (frozen dataclass)

In `hset_state.py`. The three hashes: `state_dict_hash`, `model_file_hash`, `config_hash`. Computed once per invocation by `compute_signals`.

### 4.3 `VersionInfo` and `HsetInfo` (frozen dataclasses)

In `hset_state.py`. The resolved location of the current run, with relevant metadata. Returned by `resolve_version` and `resolve_hset`.

### 4.4 `TodoSentinel`

In `todo_yaml.py`. A class whose `__int__`, `__float__`, `__bool__`, `__index__`, `__str__` all raise `NotImplementedError`. Registered as the constructor for `!todo` in PyYAML's SafeLoader.

---

## 5. The Lightning Integration Seam

ft4 uses `LightningCLI` from `lightning.pytorch.cli` — but not as a subclass and not in "run mode."

```python
# In main.py:
cli = LightningCLI(
    model_class=model_class,                       # discovered from the file
    run=False,                                     # don't have LightningCLI dispatch
    save_config_callback=None,                     # we save config ourselves
    subclass_mode_data=True,                       # allow data: class_path: ... syntax
    args=[...],                                    # cascaded args (defaults + model.yaml + CLI)
    parser_kwargs={"default_config_files": [...]}, # for !todo
)
```

After construction, `cli.model`, `cli.datamodule`, `cli.trainer`, `cli.config`, `cli.parser` are all available. ft4 inspects `cli.config.as_dict()` for signal computation and mutates `cli.trainer` post-construction via `inject_paths`.

### 5.1 The path-injection seam

`inject_paths(trainer, hset_dir, session_dir)` in `hset_state.py` is the *only* Lightning-aware bit of hset_state. It mutates the live trainer:

- `trainer.callbacks[i].dirpath` for `ModelCheckpoint` → `<hset>/checkpoints/`
- `trainer.logger._save_dir` → `<session>/`
- Also duck-types `hset_dir` and `session_dir` onto every callback (the sample callback and session finalization callback both read these).

The duck-typing pattern lets the function avoid `isinstance` checks against specific callback classes — any callback that happens to have a `hset_dir` attribute will pick up the injection.

### 5.2 Why not subclass LightningCLI

The original spec floated three implementation paths (A subclass, B direct, C subprocess). We went with B because:
- Smart-flow phase ordering (stats display BEFORE training) doesn't align with `LightningCLI`'s hooks.
- B keeps `main.py` simple and explicit; the LightningCLI is just a tool we call.
- Subclassing forces us to inherit `LightningCLI`'s argparse behavior, which we already replaced with our own argparse for the ft4-specific flags.

---

## 6. Callbacks (the two ft4-specific ones)

### 6.1 `SampleGenerationCallback`

In `sample_callback.py`. Periodically writes `<hset>/samples/step_NNNNNN.md` during training.

**Hooks:**
- `on_validation_epoch_end`: writes one sample. Skipped during `trainer.sanity_checking`.
- `on_train_batch_end`: writes one sample if `step_interval > 0` and `global_step % step_interval == 0` (and not step 0, which is owned by the phase-1 baseline).

**State injected post-construction:**
- `self.hset_dir`: set by `inject_paths`.
- `self.prompts`: set by `run.py` from `Ft4Args.prompt`.

**Contract with the model:** if `pl_module` has a `generate_samples(prompts=None) -> str` method, the callback calls it and writes the returned markdown. If absent, no-op.

**Sanity guard:** Lightning runs a ~2-batch validation at the start of `fit()` as a sanity check. Sample generation on a randomly-initialized model is noise, so the callback returns early when `trainer.sanity_checking is True`.

**Counter:** `self._val_count` increments per real (non-sanity) validation. `epoch_interval=2` fires on calls 2, 4, 6, … (not raw `trainer.current_epoch`, which doesn't advance when `val_check_interval < 1.0` triggers mid-epoch validation).

### 6.2 `SessionFinalizationCallback`

In `session_finalization.py`. The critical callback for ft4's session-record contract.

**Hooks:**
- `on_exception(trainer, pl_module, exception)`: records `status` as `"interrupted"` (for KeyboardInterrupt) or `"error"` (otherwise, with type/message in the `error` field).
- `on_fit_end`: writes the session record with `status="completed"` if no exception fired.
- `teardown(trainer, pl_module, stage)`: safety net for stage=`"fit"`. Idempotent via `_finalized` flag.

**State injected post-construction:**
- `self.hset_dir`, `self.session_dir`: set by `inject_paths`.
- `self.started_at`, `self.git_commit`, `self.git_dirty`: set by `run.py` via `start(...)` immediately before `trainer.fit()`.

**Why this callback owns both responsibilities:** the session record (sessions.jsonl) and the cumulative metrics rebuild are conceptually different (run metadata vs training data), but they share a trigger ("fit ended, however it ended"). One callback fires on three Lightning hooks; the alternative would be two callbacks each duplicating the hook handling.

**Lazy import of `_build_session_record`:** `session_finalization.py` and `run.py` would otherwise have a circular import. The callback imports `from mlops.run import _build_session_record` inside its `_finalize` method.

**No try/except in run.py:** an important architectural decision. Originally we wrapped `trainer.fit()` in try/finally to handle Ctrl-C. The current design lets Lightning's own graceful-shutdown sequence handle the interrupt, and ft4's bookkeeping rides Lightning's hooks. This is more compositional and avoids fighting Lightning's existing machinery.

---

## 7. The Config Cascade

```
defaults.yaml  →  <model>.yaml  →  CLI overrides
   (system)        (model)         (invocation)
```

Built in `main.py::build_cascade_args(ft4_args)`:

1. Read `defaults.yaml` via `_defaults_yaml_path()` (uses `__file__` to locate the package's defaults.yaml — works in both editable installs and wheel installs because `package-data` is declared in pyproject.toml).
2. If `<model_path>.yaml` exists, layer on top.
3. Append `lightning_args` from `Ft4Args` (the `--trainer.X=Y` flags etc.).

Returns a list of args strings that LightningCLI consumes via `args=...`.

The cascade is *args*, not a merged dict, because jsonargparse handles the merging natively (later args override earlier). No ft4 logic needs to know about merge semantics.

---

## 8. Hashing Discipline

### 8.1 `hash_state_dict(model)`

Sorted `(name, tuple(shape), str(dtype))` tuples from `model.state_dict()`, JSON-encoded, SHA-256. Includes buffers. Sort-by-key guarantees determinism across Python runs.

Subtle: non-persistent buffers (`register_buffer(..., persistent=False)`) are NOT in `state_dict()`, so they don't contribute to the hash. This is intentional and powers the hset-fork-vs-version-fork distinction for toggles like `use_pos_enc` (whose PE buffer is non-persistent).

### 8.2 `hash_file(path)`

SHA-256 of file bytes. Cheap. Trivially deterministic.

### 8.3 `hash_config(config_dict, exclude_prefixes)`

Flatten dict to dotted-key form (`trainer.callbacks.0.class_path`), filter by prefix match against `CONFIG_HASH_EXCLUDED_PREFIXES`, sort by key, JSON-encode with `default=str`, SHA-256.

The `default=str` fallback handles any non-JSON-serializable value (Path objects, enums) deterministically. List items are kept in order (so `[A, B]` ≠ `[B, A]`); dict keys are sorted (so `{"a":1, "b":2}` == `{"b":2, "a":1}`).

---

## 9. Filesystem Layout (Internal)

Under `runs/<model_stem>/v<NNN>/t<NNN>/`:

| Path | Owner | Written when |
|------|-------|--------------|
| `version.json` | `hset_state._create_new_version` | New version (signals mismatch latest) |
| `model.py.snapshot` | `hset_state._create_new_version` | New version |
| `hset.json` | `hset_state._create_new_hset` (config_hash); `update_hset_last_session` (last_session_at) | New hset; every session end |
| `config.yaml` | `run.persist_config` | Phase 2 start |
| `sessions.jsonl` | `hset_state.append_session` (via SessionFinalizationCallback) | Session end |
| `metrics.csv` | `cumulative_metrics.rebuild_cumulative_metrics` | Session end |
| `checkpoints/last.ckpt` | Lightning's ModelCheckpoint | During fit |
| `checkpoints/step={N}.ckpt` | Lightning's ModelCheckpoint | Per checkpoint event |
| `sessions/sNNN/metrics.csv` | Lightning's CSVLogger | During fit |
| `sessions/sNNN/hparams.yaml` | Lightning's CSVLogger | First log call |
| `samples/step_NNNNNN.md` | `sample_callback.write_sample_file` | Phase-1 baseline + per-validation |

The `sNNN/` directories are created by `hset_state.create_session_dir` immediately before `inject_paths` (which then mutates the logger to point there).

Atomic writes for cumulative metrics: `metrics.csv.tmp` → `rename`. Prevents partial files on crash.

---

## 10. Testing Layout

```
tests/
├── contract/           # 7 tests pinning Lightning's behavior we rely on
│   └── test_lightning_assumptions.py
├── unit/               # pure-logic unit tests
│   └── test_hset_state.py
├── cli/                # cli-aware unit tests (mocked Lightning)
│   ├── test_main.py
│   ├── test_run.py
│   ├── test_sample_callback.py
│   ├── test_session_finalization.py
│   ├── test_cumulative_metrics.py
│   ├── test_briefing_display.py
│   ├── test_prompts.py
│   ├── test_todo_yaml.py
│   ├── test_transformer_toggles.py
│   └── test_iris.py
└── integration/        # end-to-end through real Lightning + real fit()
    ├── test_smoke_fit.py
    ├── test_run_smart_flow.py
    ├── test_run_smart_flow_generative.py
    └── test_transformer_hset_forks_smart_flow.py
```

Three test categories with distinct purposes:

- **Contract tests** pin assumptions about Lightning's internal behavior (e.g., that `ModelCheckpoint.dirpath` can be mutated post-construction). If Lightning changes underneath us, contract tests break first and tell us exactly which assumption broke. ~7 tests.
- **Unit tests** cover Lightning-free modules in isolation. Heavy use of mocks and small data fixtures. ~120 tests.
- **Integration tests** drive real fit() loops with tiny synthetic models. ~15 tests. Slow but high-confidence.

Total ~220 tests, 80% pass time under 10 seconds.

**Important pattern: test fixture YAMLs.** Integration tests use inline `DEFAULTS_YAML` strings written to `tmp_path/defaults.yaml`, then `monkeypatch.setattr(main_mod, "_defaults_yaml_path", lambda: defaults)`. The fixture YAMLs must include the `SessionFinalizationCallback`; if you add a required callback to the production defaults.yaml, you must update the fixture YAMLs too. This is a known coupling.

---

## 11. Design Principles

These emerged during implementation and shape current decisions.

### 11.1 Don't fight Lightning's existing machinery

The first attempt at handling Ctrl-C wrapped `trainer.fit()` in try/finally. The correct design uses Lightning's `on_exception` / `on_fit_end` / `teardown` hooks. Whenever ft4 needs to react to "fit ended" or "training event happened," look for a Lightning hook first.

### 11.2 Phase 0 is read-only

The split between Phase 0 (read-only setup) and Phase 2 (destructive) is a hard line. Code that mutates the filesystem must be in Phase 2. Code that reads is in Phase 0. Exception: creating `runs/<model>/v<NNN>/` for a new version happens during resolution, but it's idempotent and intentional.

### 11.3 Duck-type, don't isinstance-check

`inject_paths` sets `hset_dir` on every callback that has the attribute. The sample callback and session finalization callback both opt in by declaring the attribute. New callbacks just need to add `self.hset_dir = None` to `__init__`.

Similarly, the `prompts` injection in `run.py` ducks on the `prompts` attribute. Future callbacks that want CLI prompts get them for free.

### 11.4 Hset-level metrics aggregation is CSV-specific

Don't try to generalize aggregation across loggers. TB has its own multi-run UI; WandB has its own. ft4's `metrics.csv` aggregation exists because CSV is the one format with no native multi-run viewer.

### 11.5 Config hash captures *scientific* configuration

`CONFIG_HASH_EXCLUDED_PREFIXES` is curated, not introspective. The list is the design. When adding a new field to exclude, add it to the list with a comment explaining why.

### 11.6 Test files are coupled to source format

Tests pinning format details (e.g., the `session` column is integer-string `"1"` in CSV, not `"s001"`) are unavoidable. When changing a format, the diff is source + every test that asserts on the format. There is no architectural fix; just discipline in the delivery.

### 11.7 KISS: prefer functions to callbacks, callbacks to subclasses

`cumulative_metrics.py` is a function, not a callback (no Lightning dependency at module level). When live updates were tried via callback, the resulting fragility (conditional class definition) led us back to the function-only design.

Lightning callbacks compose; subclassing Lightning classes (CSVLogger, LightningCLI) does not. Avoid subclassing.

---

## 12. Adding a New Subcommand

To add e.g. `ft4 list`:

1. **`main.py::_build_parser`**: add a subcommand parser with its specific flags.
2. **`main.py::_dispatch`**: route to your new function.
3. **New module under `mlops/src/mlops/`** (or extend an existing one) for the function.
4. **Use existing helpers**: `discover_model_class`, `build_lightning_cli` (if you need a model; you probably don't for `list`), `resolve_version`, `resolve_hset`, `read_sessions`.
5. **Tests**: at least one CLI unit test mocking the underlying state, and ideally an integration test that creates a few hsets then lists them.

For read-only subcommands, Phase 0 is most of your work; Phase 2 may be empty.

---

## 13. Common Pitfalls

- **Forgetting `--break-system-packages` on pip install.** ft4 uses `uv` primarily; pip support is documented but not the default. On systems with PEP 668 enforcement, plain `pip install -e .` fails.

- **`generate_samples` import cycle.** The transformer's `generate_samples` imports `LangGen` lazily inside the method. Eager import at module top would create a cycle through `ft4.lang_gen`.

- **Module shadowing in `discover_model_class`.** The discovery code registers the model file under its natural stem in `sys.modules` (so jsonargparse can resolve sibling class_paths like `iris.IrisDataModule`). If a student names their model file `lightning.py` or `torch.py`, this will break. No current check.

- **`SessionFinalizationCallback` is required.** If a `<model>.yaml` overrides the callbacks list, it must include `mlops.session_finalization.SessionFinalizationCallback`, or sessions.jsonl won't be written and hset-level metrics won't be aggregated.

- **Path injection assumes specific callback types.** `inject_paths` looks for `ModelCheckpoint` by class match (not duck-type) when setting `dirpath`. Multiple ModelCheckpoints in the callbacks list will all get the same injection (probably desired). Future custom checkpoint callbacks won't get injected unless they're a `ModelCheckpoint` subclass.

---

## 14. Where to Look for Specific Behavior

| Question | Look in |
|----------|---------|
| How are signals computed? | `hset_state.py` → `compute_signals`, `hash_*` |
| How does ft4 decide version vs hset fork? | `hset_state.py` → `resolve_version`, `resolve_hset` |
| What gets written to sessions.jsonl? | `run.py` → `_build_session_record`; `session_finalization.py` → `_finalize` |
| How are samples written? | `sample_callback.py` (per-validation); `run.py` Phase 2 (phase-1 baseline) |
| How is config_hash computed? | `hset_state.py` → `hash_config`, `CONFIG_HASH_EXCLUDED_PREFIXES` |
| Where does Lightning's logger get its path? | `hset_state.py` → `inject_paths` (mutates `logger._save_dir`) |
| How does Ctrl-C handling work? | `session_finalization.py` → `on_exception` |
| How does the !todo sentinel work? | `todo_yaml.py` → `TodoSentinel`, registered via `register_constructor` in `main.py` |
| Where's the entry point? | `pyproject.toml` `[project.scripts]` → `mlops.main:main` |
