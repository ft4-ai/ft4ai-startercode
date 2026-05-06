import os
from typing import Literal
import functools

from torch.utils.data import DataLoader
import lightning as L
import datasets

from hardcoregenai.pipeline.bpe_tokenizer import HcgaiTokenizer
from hardcoregenai.pipeline.binpack import binpack

# TODO Remove the bos/eos params; instead, take them from the tokenizer
# TODO Changing the `tokens.csv` should invalidate the cache. Perhaps add the md5(tokens.csv) to the fingerprint.

class StoriesDataModule(L.LightningDataModule):
    DATASET_ID = 'lennart-finke/SimpleStories'

    def __init__(self, seq_len: int, batch_size: int, data_size: Literal['full', 'medium', 'small']='full'):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_size = data_size
        self.num_proc = max(1, os.cpu_count() - 2) # Set to 1 for debugging # type:ignore
        print(f'{self} loaded')

    def prepare_data(self):
        splits = {}
        splits['full'] = {
            'train':      'train[:98%]',
            'validation': 'train[98%:]',
            'test':       'test'
        }
        splits['medium'] =  {
            'train':      'train[:262144]',
            'validation': 'train[-16384:]',
            'test':       'test'
        }
        splits['small'] = {
            'train':      'train[:32768]',
            'validation': 'train[-4096:]',
            'test':       'test[:4096]'
        }
        self.story_datasets = datasets.load_dataset(self.DATASET_ID, split=splits[self.data_size], num_proc=self.num_proc)

    def setup(self, stage):
        tokenizer_fn = functools.partial(_tokenize, bos_token_id=HcgaiTokenizer.RES_START, eos_token_id=HcgaiTokenizer.RES_STOP, tokenizer=HcgaiTokenizer)
        tokenized = self.story_datasets.map(function=tokenizer_fn, input_columns=['story'], batched=False, remove_columns=self.story_datasets['train'].column_names, desc='Tokenizing', num_proc=self.num_proc) # type:ignore

        # Simple implementation of https://www.amazon.science/blog/improving-llm-pretraining-with-better-data-organization
        # Also see HuggingFace chunking / group_by_texts code
        
        # Rough heuristic to determine the number of records to chunk together
        # We sample the _first_ records to keep this deterministic (and avoid breaking the fingerprint)
        statistical_sample = tokenized['train'].select(range(min(10_000, len(tokenized['train'])))) # type:ignore
        seq_lens = [len(r['tokens']) for r in statistical_sample] # type:ignore
        avg_seq_len = sum(seq_lens)/len(seq_lens) # type:ignore
        target_num_bins = 8192
        chunking_batch_size = min(8192, int(target_num_bins * self.seq_len / avg_seq_len))
        print(f'Using {self.seq_len=} {avg_seq_len=} {chunking_batch_size=}')
        
        chunking_fn = functools.partial(_chunked, seq_len=self.seq_len)
        self.chunked = tokenized.map(function=chunking_fn, input_columns=['tokens'], batched=True, batch_size=chunking_batch_size, desc='Chunking and packing', num_proc=self.num_proc)
        self.chunked.set_format(type='torch', columns=['tokens'])

    def train_dataloader(self, shuffle=True):
        return self.dataloader(self.chunked['train'], shuffle=shuffle) # type:ignore
    
    def val_dataloader(self):
        return self.dataloader(self.chunked['validation'], shuffle=False) # type:ignore
    
    def test_dataloader(self):
        return self.dataloader(self.chunked['test'], shuffle=False) # type:ignore
    
    def dataloader(self, dataset, shuffle=True):
        num_workers = self.num_proc if self.num_proc > 1 else 0 # Set to 0 for debug
        return DataLoader(dataset=dataset, batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=shuffle, num_workers=num_workers, persistent_workers=False, in_order=False)
    
    def __str__(self) -> str:
        return f'StoriesDataModule(batch_size={self.batch_size} seq_len={self.seq_len} data_size={self.data_size})'


# These helpers need to be global (i.e. outside the class)
def _tokenize(txt, bos_token_id, eos_token_id, tokenizer):
     return {'tokens': [bos_token_id] + tokenizer.tokenize(txt) + [eos_token_id]}

def _chunked(batch, seq_len):
     return {'tokens': binpack(batch, seq_len, drop_last=True)}


if __name__ == '__main__' and True:
    import torch

    B = 4
    L = 128
    
    print(f'# Demonstrating StoriesDataModule({B=} {L=})')
    print(f'## We use data_size="small" for dev, and "medium" or "full" for power')
    sdm = StoriesDataModule(batch_size=B, seq_len=L, data_size='small')
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

        print(f'\n## Tokens map words, or word fragments, to integers:\n     {HcgaiTokenizer.tokenize("the")=}\n     {HcgaiTokenizer.detokenize([4161])=}')
        print(f'\n## Metatokens such as PAD ({HcgaiTokenizer.RES_PAD}), which indicates padding, \n## or BOS ({HcgaiTokenizer.RES_START}), which indicates "beginning of sequence", are reserved.')
        print('## Our tokenizer ensures that *each* sequence is a full L tokens. \n## Other tokenizers use a special PAD token to do this.\n')

        print("## Sequences and batches are shuffled for best performance, \n## so batch[0] could be a sequence of L tokens *anywhere* in the dataset.\n")

        print(f"## Here's the first 16 tokens of the first seq of our first batch:")
        print(f'      {batch["tokens"][0,0:16]=}')
        print('## We can roundtrip tokens back to text:')
        print(f'{HcgaiTokenizer.detokenize(batch["tokens"][0,0:16])=}')
        break # break after first batch, enough for this demo

    print('\n# Experiment with StoriesDataModule until you are comfortable using it.')