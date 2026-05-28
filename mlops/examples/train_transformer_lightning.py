"""Train a small Transformer with Lightning's Trainer (no ft4 CLI).

Mirrors train_iris_lightning.py with the bits the Transformer actually
needs: bf16-mixed precision, gradient clipping, and a smaller dim/depth +
data_size="small" for quick iteration.

    uv run python mlops/examples/train_transformer_lightning.py
"""
import lightning as L
from ft4.models.transformer import Transformer
from ft4.pipeline.stories_data_module import StoriesDataModule
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

dm = StoriesDataModule(seq_len=256, batch_size=64, data_size="small")
model = Transformer(vocab_size=Ft4Tokenizer.vocab_size(), dim=128, depth=4)

trainer = L.Trainer(max_epochs=2, precision="bf16-mixed", gradient_clip_val=1.0)
trainer.fit(model, datamodule=dm)
