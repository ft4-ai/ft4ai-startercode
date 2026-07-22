from typing import Tuple, Optional, List

import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
from datasets import load_dataset, ClassLabel

from ft4.models.mlp import MlpBlock

FEATURE_COLS: List[str] = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
NUM_FEATURES = len(FEATURE_COLS)
SPECIES: List[str] = ["setosa", "versicolor", "virginica"]
NUM_SPECIES = len(SPECIES)

# Train this model: ft4 train path/to/model.py
# See mlops/README.md for details.
class IrisClassifier(L.LightningModule):
    """
    A tiny "hello world" classifier of the Iris dataset.

    """

    # A classifier, not a language model: `ft4 train` shouldn't try to
    # generate text samples from it.
    is_language_model = False

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
        self.log("train_loss", loss, prog_bar=True)
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
        self.log("val_loss", loss, prog_bar=True)
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
        # In AI, tensors' first dim is the batch size B. Even if we have only a single datum, 
        # we put it in a batch of size 1, so interfaces stay uniform:
        #   iris_measurements has shape (B, F)
        #   where B = batch_size
        #         F = num_features
        iris_measurements = ... # TODO-LAB unit1.lab0 Create the appropriate tensor
        # Hint 1: torch.tensor and torch.unsqueeze may be useful
        # Hint 2: self.device tells you the device (e.g. cpu, cuda) this model is on
        # Hint 3: See https://ft4.ai/course/articles/tensors-and-devices/
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


