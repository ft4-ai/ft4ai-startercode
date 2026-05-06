import os

from torch.utils.data import DataLoader
import lightning as L
import datasets

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

# TODO Get rid of this entirely; fold it into the regular StoriesDataModule,
# refactoring, if needed, all the models to be able to use it.

class NonChunkedStoriesDataModule(L.LightningDataModule):
    DATASET_ID = 'lennart-finke/SimpleStories'

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        split = {
            'train':      'train[:98%]',
            'validation': 'train[98%:]',
            'test':       'test'
        }
        self.story_datasets = datasets.load_dataset(self.DATASET_ID, split=split, num_proc=4) # type:ignore

    def setup(self, stage):
        def _tokenizer(txt):
            return {'tokens': [Ft4Tokenizer.RES_START] + Ft4Tokenizer.tokenize(txt) + [Ft4Tokenizer.RES_STOP]} # type:ignore

        tokenized = self.story_datasets.map(function=_tokenizer, input_columns=['story'], batched=False, remove_columns=self.story_datasets['train'].column_names, desc='Tokenizing', num_proc=os.cpu_count()) # type:ignore

        def _add_sort_key(record, idx):
            return {'sort_key': -len(record['tokens']), 'orig_idx': idx} # Preserve orig_idx for stable sorting
        
        # Sort by len(tokenized), so that stories of similar len batch together, reducing padding
        self.tokenized_sorted_by_len = tokenized.map(_add_sort_key, with_indices=True, desc='Sorting by num. tokens').sort(['sort_key', 'orig_idx']).remove_columns(['sort_key', 'orig_idx'])
        self.tokenized_sorted_by_len.set_format('torch')

    def train_dataloader(self):
        return self.dataloader(self.tokenized_sorted_by_len['train']) # type:ignore
    
    def val_dataloader(self):
        return self.dataloader(self.tokenized_sorted_by_len['validation']) # type:ignore
    
    def test_dataloader(self):
        return self.dataloader(self.tokenized_sorted_by_len['test']) # type:ignore
    
    def dataloader(self, dataset):
        num_workers = max(1, os.cpu_count() // 2) # type:ignore
        # For now, we do no collation
        return DataLoader(dataset=dataset, batch_size=self.batch_size, pin_memory=False, drop_last=True, num_workers=num_workers, in_order=False)

if __name__ == '__main__':
    sdm = NonChunkedStoriesDataModule(batch_size=2)
    sdm.prepare_data()
    sdm.setup('train')
    print(sdm.train_dataloader())
