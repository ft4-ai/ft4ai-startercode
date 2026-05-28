"""Load a saved Transformer checkpoint and generate text with LangGen.

A .ckpt file produced by Lightning contains both the model weights and the
__init__ hyperparameters (vocab_size, dim, depth, ...) saved by the model's
self.save_hyperparameters() call. So `Model.load_from_checkpoint(path)`
rebuilds the model from a single file — no separate hparams.yaml needed.

The checkpoint path below assumes you've previously run:
    ft4 train src/ft4/models/transformer.py

Run from the project root:
    uv run python mlops/examples/load_and_generate.py
"""
import torch

from ft4.models.transformer import Transformer
from ft4.lang_gen import LangGen

# Randomize sampling each run; LangGen uses torch.multinomial which respects
# the global RNG, so without a fresh seed every run produces identical output.
torch.manual_seed(torch.seed())

CKPT_PATH = "runs/transformer/v001/h001/checkpoints/last.ckpt"
PROMPT = "Once upon a time"

model = Transformer.load_from_checkpoint(CKPT_PATH)
model.eval()

print(PROMPT, end="", flush=True)
for token in LangGen(model=model, initial_txt=PROMPT, max_to_generate=128):
    print(token, end="", flush=True)
print()
