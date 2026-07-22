"""SampleGenerationCallback.

Periodically calls the ft4 sampler to generate samples during training and
writes results to `<hset>/samples/step_NNNNNN.md`.

For models that work with LangGen (token-in, logits-out language models),
samples are generated automatically. For non-generative models (classifiers
like iris, or anything LangGen can't drive), generation raises — we swallow
it, print a one-time warning on the first failure, and stay silent after.

Step numbers in filenames are global (trainer.global_step), so they correctly
sequence across resumed sessions: session 1 might write step_000050.md
through step_001000.md, and session 2 then continues from step_001050.md.
Phase 1 baseline samples (written by run.py before training) use step 0.

Two independent triggers via init args:
  - epoch_interval: write every N validation epochs (0 = never). Note
    "validation epoch" not "training epoch": with val_check_interval < 1.0,
    validation runs multiple times per training epoch, and the callback
    fires after each one. Lightning's sanity-check validation at the
    start of fit() is skipped.
  - step_interval:  write every N training steps (0 = never).

Defaults: epoch_interval=1, step_interval=0 — one sample file per validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import lightning as L

class SampleGenerationCallback(L.Callback):
    """Periodic sample-generation callback. Silently no-ops (with a single
    stderr warning on the first failure) if the model can't be driven by
    LangGen."""

    def __init__(
        self,
        epoch_interval: int = 1,
        step_interval: int = 0,
    ) -> None:
        super().__init__()
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        # Set post-instantiation by hset_state.inject_paths and run.py:
        self.hset_dir: Optional[Path] = None
        self.prompts: Optional[list[str]] = None
        # Counts validation epochs (not training epochs); used to evaluate
        # `epoch_interval`. With val_check_interval < 1.0 in trainer config,
        # validation runs many times per training epoch — using a separate
        # counter keeps "interval=1 fires every validation" honest.
        self._val_count = 0
        self._failure_warned = False

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking or getattr(pl_module, "has_optimizer", True):
            return
        # The empirical model has no optimizer, and therefore leaves global_step
        # stuck 0 (since it reads from manual_optimization.optim_step_progress, which only advances when an
        # optimizer.step() runs). So we sync to total_batch_idx so ModelCheckpoint's
        # duplicate-save guard doesn't fire and filename templates show real
        # progress. This relies on Lightning's internals, which seem stable in 2.x.
        progress = trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress
        progress.total.completed = trainer.fit_loop.epoch_loop.total_batch_idx

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Lightning runs a sanity validation at the start of fit() (by
        # default ~2 batches); samples from a randomly-initialized model
        # are noise, so skip those.
        if trainer.sanity_checking:
            return
        if self.epoch_interval <= 0:
            return
        self._val_count += 1
        if self._val_count % self.epoch_interval != 0:
            return
        self._maybe_write(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.step_interval <= 0:
            return
        step = int(trainer.global_step)
        if step == 0:
            return  # phase 1 baseline owns step 0
        if step % self.step_interval != 0:
            return
        self._maybe_write(trainer, pl_module)

    def _maybe_write(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.hset_dir is None:
            return  # not yet injected; shouldn't happen if smart flow ran
        from mlops.sampling import (
            SamplingUnsupported,
            generate_sample_markdown,
            write_sample_file,
        )
        if self._failure_warned:
            return  # already skipped/failed once this session; don't retry
        step = int(trainer.global_step)
        try:
            text = generate_sample_markdown(
                pl_module,
                prompts=self.prompts,
                step=step,
                max_to_generate=40,
            )
            write_sample_file(self.hset_dir, step, text)
            # Stash the latest sample on the module so LiveSampleCallback
            # (and any other consumer) can pick it up without duplicating
            # the generation work.
            pl_module._ft4_latest_sample = (step, text) #type:ignore
        except SamplingUnsupported:
            # Model declared itself non-generative (e.g. a classifier like
            # iris). Expected and uninteresting: skip silently this session.
            self._failure_warned = True
        except Exception as e:
            self._failure_warned = True
            print(
                f"Warning: this model was expected to generate samples, but "
                f"sampling failed; skipping samples for the rest of this "
                f"session. If this model isn't a language model, set "
                f"is_language_model = False on it to silence this. "
                f"Underlying error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
