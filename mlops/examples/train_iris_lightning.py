"""Train IrisClassifier with Lightning's Trainer (no ft4 CLI).

The "Hello, World" of Lightning training: instantiate the model and
datamodule, build a Trainer, call fit. Lightning handles dataloaders,
device placement, the train/val loop, and metric logging.

    uv run python mlops/examples/train_iris_lightning.py
"""
import lightning as L
from ft4.models.iris import IrisClassifier, IrisDataModule

model = IrisClassifier()
dm = IrisDataModule(batch_size=32)

trainer = L.Trainer(max_epochs=32)
trainer.fit(model, datamodule=dm)
