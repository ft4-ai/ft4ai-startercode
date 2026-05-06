import sys
import warnings

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from hardcoregenai.models.empirical_model import EmpiricalModel
from hardcoregenai.app.language_model import LangGen
from hardcoregenai.pipeline.stories_data_module import StoriesDataModule

# TODO Add an option to load a saved ckpt and skip training

# Silence unnneeded warnings
warnings.filterwarnings("ignore", message=r".*configure_optimizers.*None.*")
warnings.filterwarnings("ignore", message=r".*GPU available but not used.*", category=UserWarning)
torch.set_float32_matmul_precision('medium') # Silence warnings

# TODO-LAB unit1.lab1
# Experiment with different values of these hyperparameters
raise NotImplementedError("Tune hyperparameters - unit1.lab1")
N = 1 # TODO-LAB What size n-gram? This determines the model's behavior, power, and performance.
DATA_SIZE = "small" # Start with "small" for dev, then change to "medium" or "full" for power
BATCH_SIZE = 512 # Tune for max speed; you should be able to make this quite large

SEQ_LEN = 512 # Adjust if needed
# This model's validation is atypically memory hungry and slow, 
# so we use smaller and fewer batches.
VAL_BATCH_SIZE = 32
LIMIT_VAL_BATCHES = 16 # Limit to a subset of the dataset

CKPT_DIR = "./checkpoints/empirical_model"
logger = CSVLogger(save_dir=CKPT_DIR)
ckpt_cb = ModelCheckpoint(
    dirpath=logger.log_dir,
    save_on_train_epoch_end=True,
    save_last='link',
)

model = EmpiricalModel(n=N)
print(f'# {model}: {DATA_SIZE=} {BATCH_SIZE=}')

# Train
print('# Training...')
datamodule = StoriesDataModule(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, data_size=DATA_SIZE)
trainer = L.Trainer(
    devices=1,
    max_epochs=1,   
    check_val_every_n_epoch=0,
    limit_val_batches=0,
    callbacks=[ckpt_cb, RichProgressBar(leave=True)],
    logger=logger,
    enable_model_summary=False,
)
trainer.fit(model=model, datamodule=datamodule)

# For performance reasons, we validate separately
print('# Validating...')
val_datamodule=StoriesDataModule(seq_len=SEQ_LEN, batch_size=VAL_BATCH_SIZE, data_size=DATA_SIZE)
val_trainer = L.Trainer(
    accelerator='cpu', # Due to its dict, validation runs faster on the CPU
    limit_val_batches=LIMIT_VAL_BATCHES,
    num_sanity_val_steps=0,
    callbacks=[ModelCheckpoint(), RichProgressBar(leave=True)],
    logger=logger,
)
val_trainer.validate(model=model, datamodule=val_datamodule)

# And generate!
PROMPT = "Now, she wondered if"
print('# Generating...')
model.to('cpu')
model.eval() # Setting eval mode improves performance of some models
for i in range(2):
    response = LangGen(model=model, initial_txt=PROMPT, max_to_generate=128)

    sys.stdout.write(f"## Empirical Model n={model.n}:\n\n")
    sys.stdout.write(LangGen.bold(PROMPT))
    sys.stdout.flush()
    for txt in response:
        sys.stdout.write(txt)
        sys.stdout.flush()
    sys.stdout.write("\n\n")
    # TODO Save the generated story next to metrics.csv
