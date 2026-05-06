from typing import Tuple, Optional, List

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
from datasets import load_dataset, ClassLabel

from hardcoregenai.models.mlp import MlpBlock

FEATURE_COLS: List[str] = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
NUM_FEATURES = len(FEATURE_COLS)
SPECIES: List[str] = ["setosa", "versicolor", "virginica"]
NUM_SPECIES = len(SPECIES)

class IrisClassifier(L.LightningModule):
    """
    A tiny "hello world" classifier of the Iris dataset.

    """

    def __init__(self) -> None:
        assert NUM_FEATURES == 4
        assert NUM_SPECIES == 3
        H_DIM = 16
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, H_DIM),
            MlpBlock(H_DIM),
            nn.Linear(H_DIM, NUM_SPECIES),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, iris_measurements: Tensor) -> Tensor:
        """   
        iris_measurements: sepal_len, sepal_width, petal_len, petal_width
                           shape (B, NUM_FEATURES) float32
        returns:           logits
                           shape (B, NUM_SPECIES)  float32
        """
        return self.net(iris_measurements)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        batch:
            iris_measurements, shape (B, NUM_FEATURES) float32
            species,           shape (B,)              int64
        returns:
            loss: CE, scalar tensor
        """
        iris_measurements, species = batch
        logits = self(iris_measurements)
        loss = self.criterion(logits, species)
        self.log("train_ce", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        batch:
            iris_measurements, shape (B, NUM_FEATURES) float32
            species,           shape (B,)              int64
        """
        iris_measurements, species = batch
        logits = self(iris_measurements)
        preds = logits.argmax(dim=-1)
        acc = (preds == species).float().mean()
        loss = self.criterion(logits, species)
        self.log("val_ce", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=0.0)

    @torch.inference_mode()
    def classify_iris(
        self,
        sepal_len: float,
        sepal_width: float,
        petal_len: float,
        petal_width: float,
    ) -> int:
        """
        Returns the predicted class id (0=setosa, 1=versicolor, 2=virginica)
        for a single flower.

        Args:
            sepal_len, sepal_width, petal_len, petal_width: Python floats.
        Returns:
            cls_id: int in {0,1,2}.
        """
        # Wrap the data into a tensor of shape (1,4) and place it on this model's device.
        # In AI, tensors usually have a first dim == batch_size
        # Even if we're working with just a single datum, we use a batch_size of 1.
        # This keeps interfaces uniform.
        iris_measurements = None # TODO-LAB unit1.lab0 
        # Hint 1: torch.tensor and torch.unsqueeze may be useful
        # Hint 2: self.device tells you the device (e.g. cpu, cuda) this model is on
        raise NotImplementedError("Implement in unit1.lab0")
        logits = self(iris_measurements)
        return int(logits.argmax(dim=-1).item())

class IrisDataModule(L.LightningDataModule):
    """
    Iris data.
    The dataloaders return batches with:
            iris_measurements: shape (B, NUM_FEATURES) float32
                               sepal_len, sepal_width, petal_len, petal_width (in cm)
            species:           shape (B,)              int64
                               0=setosa 1=versicolor 2=virginica
    """
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_ds: Optional[TensorDataset] = None
        self.val_ds: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        ds = load_dataset("scikit-learn/iris")

        # Make 'Species' a ClassLabel (needed for stratify_by_column)
        label_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        ds = ds.cast_column("Species", ClassLabel(names=label_names))

        split = ds["train"].train_test_split(test_size=0.2, stratify_by_column="Species", seed=42)
        tr, va = split["train"], split["test"]

        def to_tensors(dset) -> TensorDataset:
            iris_measurements = torch.tensor([[float(row[c]) for c in FEATURE_COLS] for row in dset], dtype=torch.float32)
            # After cast, Species yields integer ids 0/1/2 directly
            species = torch.tensor([int(row["Species"]) for row in dset], dtype=torch.long)
            return TensorDataset(iris_measurements, species)

        self.train_ds = to_tensors(tr)
        self.val_ds = to_tensors(va)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0) #type:ignore

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0) #type:ignore


if __name__ == "__main__":
    # Demo: Train and make some example predictions

    import warnings
    from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar

    warnings.filterwarnings("ignore", message=r".*does not have many workers.*", category=UserWarning,)
    torch.set_float32_matmul_precision('medium') # Silence warnings

    print('Iris classifier demo: "hello world" of AI\n')
    dm = IrisDataModule(batch_size=32)
    dm.setup()
    model = IrisClassifier()
    trainer = L.Trainer(
        max_epochs=16,
        callbacks=[RichProgressBar()],
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    print('\nPredicting...')
    model.eval() # Switching to eval mode improves some models
    ex1 = (5.1, 3.5, 1.4, 0.2)  # likely setosa
    ex2 = (6.0, 2.9, 4.5, 1.5)  # likely versicolor
    for ex in (ex1, ex2):
        pred = model.classify_iris(*ex)
        print(f"Input {ex} -> predicted: {SPECIES[pred]} ({pred})")
