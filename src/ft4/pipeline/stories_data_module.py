import os
import functools
from typing import Literal

from torch.utils.data import DataLoader
import lightning as L
# Turn off "Warning... set a HF_TOKEN to enable higher rate limits..." ad
# This must be *before* we import datasets
os.environ["HF_HUB_VERBOSITY"] = "error"
import datasets

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer
from ft4.pipeline.binpack import binpack

# TODO Remove the bos/eos params; instead, take them from the tokenizer
# TODO Changing the `tokens.csv` should invalidate the cache. Perhaps add the md5(tokens.csv) to the fingerprint.


class StoriesDataModule(L.LightningDataModule):
    DATASET_ID = 'lennart-finke/SimpleStories'
    SHUFFLE_SEED = 0

    # Per-size split configuration. train_max / test_max of None means "use everything available".
    # Val is taken from the tail of the shuffled train split, so train and val never overlap.
    # Test is a split (of about 1%) made by the dataset's creator. We don't generally use it.
    SIZE_CONFIG = {
        'full':   {'train_max': None,   'val_size': 2048, 'test_max': None},
        'medium': {'train_max': 262144, 'val_size': 2048, 'test_max': None},
        'small':  {'train_max':  32768, 'val_size': 1024, 'test_max': 2048},
    }

    def __init__(self, seq_len: int, batch_size: int, data_size: Literal['full', 'medium', 'small']='full', val_batch_size: int | None = None):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        # val_batch_size is optional; defaults to batch_size. Useful for models whose
        # forward pass (used during validation) is more memory-hungry than training.
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.data_size = data_size
        self.num_proc = max(1, (os.cpu_count() or 4) - 2)  # Set to 1 for debugging

    def prepare_data(self):
        # Populate the local HF cache. Under DDP, prepare_data runs only on rank 0;
        # setup() re-loads from cache on every rank.
        print(f'Loading {self.DATASET_ID} (first-time download can take a while; cached after).')
        datasets.load_dataset(self.DATASET_ID, split='train', num_proc=self.num_proc)
        datasets.load_dataset(self.DATASET_ID, split='test',  num_proc=self.num_proc)

    def setup(self, stage):
        # Re-load from cache and build the split dict. Shuffle train deterministically
        # before splitting off the validation tail, so val isn't biased by file order.
        train_full = datasets.load_dataset(self.DATASET_ID, split='train', num_proc=self.num_proc).shuffle(seed=self.SHUFFLE_SEED)
        test_full  = datasets.load_dataset(self.DATASET_ID, split='test',  num_proc=self.num_proc)

        cfg = self.SIZE_CONFIG[self.data_size]
        n = len(train_full) # type:ignore
        val_size = cfg['val_size']
        train_max = cfg['train_max'] if cfg['train_max'] is not None else (n - val_size)
        val   = train_full.select(range(n - val_size, n))
        train = train_full.select(range(min(train_max, n - val_size)))
        test  = test_full if cfg['test_max'] is None else test_full.select(range(min(cfg['test_max'], len(test_full)))) # type:ignore
        self.story_datasets = datasets.DatasetDict({'train': train, 'validation': val, 'test': test})

        print('Tokenizing and chunking (first run can take a while; cached after).')
        tokenizer_fn = functools.partial(_tokenize, bos_token_id=Ft4Tokenizer.RES_START, eos_token_id=Ft4Tokenizer.RES_STOP, tokenizer=Ft4Tokenizer)
        tokenized = self.story_datasets.map(function=tokenizer_fn, input_columns=['story'], batched=False, remove_columns=self.story_datasets['train'].column_names, desc='Tokenizing', num_proc=self.num_proc) # type:ignore

        # Simple implementation of https://www.amazon.science/blog/improving-llm-pretraining-with-better-data-organization
        # Also see HuggingFace chunking / group_by_texts code

        # Rough heuristic to determine the number of records to chunk together.
        # We sample the _first_ records (post-shuffle, so representative) to keep this
        # deterministic and avoid breaking the fingerprint.
        statistical_sample = tokenized['train'].select(range(min(10_000, len(tokenized['train'])))) # type:ignore
        seq_lens = [len(r['tokens']) for r in statistical_sample] # type:ignore
        avg_seq_len = sum(seq_lens)/len(seq_lens)
        target_num_bins = 8192
        chunking_batch_size = min(8192, int(target_num_bins * self.seq_len / avg_seq_len))
        #print(f'Using {self.seq_len=} {avg_seq_len=} {chunking_batch_size=}')

        chunking_fn = functools.partial(_chunked, seq_len=self.seq_len)
        self.chunked = tokenized.map(function=chunking_fn, input_columns=['tokens'], batched=True, batch_size=chunking_batch_size, desc='Chunking and packing', num_proc=self.num_proc)
        self.chunked.set_format(type='torch', columns=['tokens'])

    def train_dataloader(self, shuffle=True):
        return self.dataloader(self.chunked['train'], batch_size=self.batch_size, shuffle=shuffle, persistent_workers=True) # type:ignore

    def val_dataloader(self):
        return self.dataloader(self.chunked['validation'], batch_size=self.val_batch_size, shuffle=False, persistent_workers=False) # type:ignore

    def test_dataloader(self):
        return self.dataloader(self.chunked['test'], batch_size=self.val_batch_size, shuffle=False, persistent_workers=False) # type:ignore

    def dataloader(self, dataset, batch_size, shuffle, persistent_workers):
        num_workers = self.num_proc if self.num_proc > 1 else 0  # Set to 0 for debug
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    def __str__(self) -> str:
        vbs = '' if self.val_batch_size == self.batch_size else f' val_batch_size={self.val_batch_size}'
        return f'StoriesDataModule(batch_size={self.batch_size}{vbs} seq_len={self.seq_len} data_size={self.data_size})'


# These helpers need to be global (i.e. outside the class)
def _tokenize(txt, bos_token_id, eos_token_id, tokenizer):
    return {'tokens': [bos_token_id] + tokenizer.tokenize(txt) + [eos_token_id]}

def _chunked(batch, seq_len):
    return {'tokens': binpack(batch, seq_len, drop_last=True)}


if __name__ == '__main__' and True:
    import torch

    BATCH_SIZE = 4
    SEQ_LEN = 128

    print(f'# Demonstrating StoriesDataModule({BATCH_SIZE=} {SEQ_LEN=})')
    print(f'## We use data_size="small" for dev, and "medium" or "full" for power')
    sdm = StoriesDataModule(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, data_size='small')
    print('\n## .prepare_data() downloads and caches the data...')
    sdm.prepare_data()
    print('## .setup("train") tokenizes it and packages it into (B,L) shaped tensors...')
    sdm.setup('train')
    print('## .train_dataloader() gives us an iterable DataLoader')
    train_dl = sdm.train_dataloader(shuffle=False)

    for batch in train_dl:
        print(f'## Each iteration returns a dict-like `batch`, with `batch[tokens]` a (B,L) tensor:')
        print(f"     {batch['tokens'].shape=}\n")
        torch.set_printoptions(threshold=3)  # show a few tokens at each end
        print(f"     batch['tokens']=\n{batch['tokens']}")

        print(f'\n## Tokens map words, or word fragments, to integers:\n     {Ft4Tokenizer.tokenize("the")=}\n     {Ft4Tokenizer.detokenize([4161])=}')
        print(f'\n## Metatokens such as PAD ({Ft4Tokenizer.RES_PAD}), which indicates padding, \n## or BOS ({Ft4Tokenizer.RES_START}), which indicates "beginning of sequence", are reserved.')
        print('## Our tokenizer ensures that *each* sequence is a full L tokens. \n## Other tokenizers use a special PAD token to do this.\n')

        print("## Sequences and batches are shuffled for best performance, \n## so batch[0] could be a sequence of L tokens *anywhere* in the dataset.\n")

        print(f"## Here's the first 16 tokens of the first seq of our first batch:")
        print(f'      {batch["tokens"][0,0:16]=}')
        print('## We can roundtrip tokens back to text:')
        print(f'{Ft4Tokenizer.detokenize(batch["tokens"][0,0:16])=}')
        break  # break after first batch, enough for this demo

    print('\n# Experiment with StoriesDataModule until you are comfortable using it.')
