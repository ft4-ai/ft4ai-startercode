# ft4 CLI runner contract tests

These tests pin Lightning integration assumptions that ft4 CLI runner V1's design relies on.
Each test fails loudly if Lightning's behavior changes in a way that breaks our
assumptions. Run once after pinning Lightning's version, re-run on Lightning
upgrades.

**NOTE:** This doc is not maintained, and the code may be ahead of it. When there's a conflict, it's the code that's correct.

## What's pinned

| Test | Contract | Consequence if it fails |
|---|---|---|
| `test_modelcheckpoint_dirpath_mutation` | `ModelCheckpoint.dirpath` can be set after `__init__` and takes effect at fit time. | `inject_paths` strategy must change; likely fallback is two-pass instantiation (instantiate model alone, route, then construct trainer with dirpath baked in). |
| `test_csvlogger_save_dir_mutation` | `CSVLogger._save_dir` is read lazily; mutation after `__init__` takes effect on first `log_metrics`. | Same as above, or subclass `CSVLogger` with sentinel resolution. |
| `test_csvlogger_cross_process_append` | A second Python process pointed at an existing CSVLogger dir appends to `metrics.csv` rather than truncating. | `TimestampedCSVLogger.__init__` opens `metrics.csv` in append mode and writes a header only if the file is empty. Small change, isolated. |
| `test_lightning_cli_run_false` | `LightningCLI(run=False, save_config_callback=None)` constructs a usable cli; `trainer.fit` works on `cli.model`/`cli.datamodule`; `cli.config` dumps reproducibly across builds (basis for stable `config_hash`). | Option (A) from spec § 4.5 is infeasible; fall back to (B) `jsonargparse` + `Trainer` directly, or (C) subprocess composition. This is the highest-stakes test. |

## Running

```bash
uv run pytest mlops/tests/contract/ -v
```

or with an activated venv:

```bash
pytest mlops/tests/contract/ -v
```

All four must pass for the V1 design path to fly. If any fail, the table above
points to the documented fallback.

## What's deliberately not here

- **`!todo` + jsonargparse type-checking interaction.** Covered later by unit
  tests on `todo_yaml.py` plus a small integration test against a model file
  with a `!todo` field. Either parse-time rejection or instantiate-time
  `NotImplementedError` is acceptable as long as the tag string surfaces.
- **`torch.compile` + `LangGen` compatibility.** Verified by a smoke test
  during `run.py` integration. The mitigation (pass uncompiled `cli.model` to
  `LangGen`; pass compiled wrapper only to `trainer.fit`) is already in the
  design.
- **`state_dict_hash` determinism.** Enforced by sorting parameter names in
  `hash_state_dict`; covered by a unit test in `tests/unit/`.
