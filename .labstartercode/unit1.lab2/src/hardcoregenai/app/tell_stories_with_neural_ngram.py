import sys
import os
import glob

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from ft4.pipeline.stories_data_module import StoriesDataModule
from ft4.app.language_model import LangGen
from ft4.models.neural_ngram import NeuralNgram

torch.set_float32_matmul_precision('medium')
# FUTURE Add an option to load a saved ckpt and skip training

TRAIN = True
CKPT_DIR = './checkpoints/neural_ngram/'

if TRAIN:
    # TODO-LAB unit1.lab2
    # Experiment with different values of these hyperparameters
    raise NotImplementedError("Tune hyperparameters - unit2.lab1")
    N = 1 # TODO-LAB What size n-gram?
    DIM = 128
    DEPTH = 8
    DATA_SIZE = "small" # Start with "small" for dev, then change to "medium" or "full" for power
    BATCH_SIZE = 64 # Tune for max speed (i.e min epoch time)
    COMPILE = False # Compiling can increase speed by up to 2x, at the cost of some startup time
    SEQ_LEN = 256
    TRAIN_TIME = '00:10:00:00', # DD:HH:MM:SS
    # TODO doc mixed precision
    logger = CSVLogger(save_dir=CKPT_DIR)
    ckpt_cb = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_last='link',
    )

    model = NeuralNgram(n = N, vocab_size=Ft4Tokenizer.vocab_size(), dim=DIM, depth=DEPTH)

    # Train
    print('# Training...')
    print('## Quick smoke test before the full training')
    trainer = L.Trainer(fast_dev_run=2, 
                        callbacks=[RichProgressBar()], 
                        logger=logger, 
                        enable_model_summary=False, 
                        enable_checkpointing=False)
    sdm = StoriesDataModule(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, data_size=DATA_SIZE)
    trainer.fit(model, datamodule=sdm) 

    if COMPILE:
        print('## Compiling...', end='')
        model = torch.compile(model)
        print(' done.')

    print('## Full training...')
    trainer = L.Trainer(precision='bf16-mixed', 
                        max_time = TRAIN_TIME,
                        gradient_clip_val=1.0,
                        default_root_dir=CKPT_DIR,
                        callbacks=[ckpt_cb, RichProgressBar(leave=True)],
                        logger=logger)
    trainer.fit(model, datamodule=sdm) #type:ignore

else:
    all_ckpts = glob.glob(os.path.join(CKPT_DIR, "**", "*.ckpt"), recursive=True)
    latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    print(f'# Loading checkpoint {latest_ckpt}')
    model = NeuralNgram.load_from_checkpoint(latest_ckpt)
    print(f'## Loaded {model}')

# And generate!
print('# Generating...')
PROMPT = ["Now, she wondered if", "Billy was very scared, so he"]
model.eval() # Setting eval mode improves performance of some models #type:ignore
for i in range(2):
    response = LangGen(model=model, initial_txt=PROMPT[i], max_to_generate=128)

    sys.stdout.write(f"## {model!s}:\n\n")
    sys.stdout.write(LangGen.bold(PROMPT[i]))
    sys.stdout.flush()
    for txt in response:
        sys.stdout.write(txt)
        sys.stdout.flush()
    sys.stdout.write("\n\n")
    # TODO Save the generated story next to metrics.csv
