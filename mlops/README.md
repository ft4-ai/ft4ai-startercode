# Training Models with `ft4` MLops

The quickest way to train your models is via the `ft4` tool:

    (.venv) $ ft4 train path/to/my/model.py

Once trained, give a prompt and generate:

    (.venv) $ ft4 generate path/to/my/model.py --prompt "Sam really wanted peanut butter"

Specify hyperparameters in `path/to/my/model.yaml` or on the command line:

    (.venv) $ ft4 train path/to/my/model.py --compile --model.dim=128 --model.depth=4 --trainer.max_epochs=5

You don't need to use `ft4` (see below), but it may save you time. 
Below, we explain what `ft4` does, why, and how it works.

## Why should I use `ft4`?

ML engineering is like software engineering from the year 2000, before `git` and `pytest`:
we struggle to track history, reliably undo changes, and produce deterministic, testable,
automated builds. Model behavior depends not only on code (which we track well), but on
data (*huge*), hyperparameters (which need to tune) and parameters (*opaque*). ML unit
testing is in its infancy.

We need to experiment—tuning code, hyperparameters, trainining runs, data, and models—and
we need to track these. We need to restore checkpoints when we can (so we don't have to
retrain from scratch), but start new ones when we must (e.g. a hyperparameter changes
incompatibly).

`ft4` is a simple tool to do exactly that. It's small enough to be included in this course, but
complete enough to automate  MLops so you can focus on the AI, not the bookkeeping. When
possible, `ft4` resumes training from your last checkpoint. But, when the model's code has
changed (`version`), or your hyperparameters have changed in an incompatible way (`hset`),
`ft4` forks and trains afresh. Set hyperparameters on the command line or in the model's `yaml`
files. Any argument your model takes in its constructor can be set via `--model.argname=`.
After training, use `ft4 list`, `ft4 show`, and `ft4 generate`.

## Why *shouldn't* I use `ft4`?

If you want more transparency and control, you 
can skip `ft4` entirely and do something like:

```python
dm = StoriesDataModule(seq_len=256, batch_size=64, data_size="small")
model = Transformer(vocab_size=Ft4Tokenizer.vocab_size(), dim=128, depth=4)
trainer = L.Trainer(max_epochs=2, precision="bf16-mixed", gradient_clip_val=1.0)
trainer.fit(model, datamodule=dm)
```
to train. Then generate: 
```python
CKPT_PATH = "runs/transformer/v001/h001/checkpoints/last.ckpt"
model = Transformer.load_from_checkpoint(CKPT_PATH)
model.eval()
PROMPT = "Sam really wanted peanut butter"
print(PROMPT, end="", flush=True)
for token in LangGen(model=model, initial_txt=PROMPT, max_to_generate=128):
    print(token, end="", flush=True)
print()
```

See `examples/` for more.

## Internals & Details

For details, see

*  `ft4 --help`
* The `yaml` files: `mlops/ft4.yaml`, `src/ft4/models/defaults.yaml`, and `path/to/model.yaml`
* Extensive unit tests: `pytest mlops/tests`

This is *your* code base, and you're encouraged to hack on it (pull requests welcome, too). That being said, the course is about *models*, not *MLops*, and so most of your time should be spent there.

## Can I use `ft4` for other AI/ML projects?

Probably—with a few changes. Forks and pull requests are encouraged, especially if you'd
like to extract `ft4` as a free-standing, open source, lightweight mini-MLops tool. Feel
free to get in touch if you have questions.

