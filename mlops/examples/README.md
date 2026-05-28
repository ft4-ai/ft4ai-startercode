# mlops/examples — "What `ft4 train` does behind the scenes"

These standalone scripts show the moving parts that the `ft4 train` CLI
normally hides. They're reference material, not part of the supported CLI
surface. Read them when you want to understand what the framework is doing,
or when you want a starting point for something the CLI doesn't (yet) cover.

Each script is meant to be run from the project root with `uv run python`.

## Scripts

- **`train_iris_lightning.py`** — the "Hello, World" of PyTorch Lightning.
  Instantiate model + datamodule, build a `Trainer`, call `fit`. ~10 lines.

- **`train_iris_explicit.py`** — the same training, but written by hand as a
  PyTorch loop (no `Trainer`). Shows exactly what `trainer.fit()` is doing:
  the train/val phase split, the zero_grad/backward/step triple, `train()` /
  `eval()` toggling, and a `no_grad()` validation pass.

- **`train_transformer_lightning.py`** — the Lightning version for a small
  Transformer: adds bf16-mixed precision and gradient clipping.

- **`train_transformer_explicit.py`** — the Transformer training written by
  hand. Highlights device placement, the next-token shift
  (`tokens[..., :-1]` → `tokens[..., 1:]`), and manual gradient clipping.

- **`load_and_generate.py`** — load a saved Transformer checkpoint and
  generate text token-by-token with `LangGen`. Demonstrates that a Lightning
  `.ckpt` carries both weights and `__init__` hparams, so
  `Model.load_from_checkpoint(path)` is all you need.

## Why these aren't part of the CLI

`ft4 train` adds experiment versioning, provenance capture, briefing display,
and sample generation on top of `trainer.fit()`. That's the right interface
for day-to-day work, but it also obscures the underlying mechanics. Keep
these scripts around for when you want to see those mechanics directly.
