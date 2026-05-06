import pytest
import logging
import torch
import lightning as L

from hardcoregenai.models.iris import IrisClassifier, IrisDataModule

L.seed_everything(42)
for name in ["lightning", "lightning.pytorch", "lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda"]:
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

@pytest.mark.lab("unit1.lab0")
def test_forward_shape() -> None:
    model = IrisClassifier().eval()
    x = torch.randn(3, 4, dtype=torch.float32)  # (B=3,4)
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (3, 3)


@pytest.mark.lab("unit1.lab0")
@pytest.mark.filterwarnings("ignore:.*worker.*")
def test_train_and_validate_threshold(iris_dm) -> None:
    model = IrisClassifier()

    trainer = L.Trainer(
        max_epochs=32,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=iris_dm)

    assert "val_acc" in trainer.callback_metrics, "val_acc not logged"
    val_acc = float(trainer.callback_metrics["val_acc"])
    assert val_acc >= 0.80, f"Expected val_acc >= 0.80, got {val_acc:.3f}"


@pytest.mark.lab("unit1.lab0")
@pytest.mark.filterwarnings("ignore:.*worker.*")
def test_classify_iris(iris_dm) -> None:
    model = IrisClassifier()
    trainer = L.Trainer(max_epochs=32,
                        enable_progress_bar=False, 
                        enable_checkpointing=False,
                        enable_model_summary=False,
                        logger=False)
    trainer.fit(model, datamodule=iris_dm)

    # Take some samples from val split
    X_val, y_val = iris_dm.val_ds.tensors  # type: ignore[attr-defined]
    correct = 0
    total = 8
    for i in range(total):
        x = X_val[i]
        sepal_len, sepal_width, petal_len, petal_width = map(float, x.tolist())
        pred = model.classify_iris(sepal_len, sepal_width, petal_len, petal_width)
        if pred == int(y_val[i].item()):
            correct += 1

    acc = correct / total
    assert acc >= 0.70, f"Expected >= 0.70 accuracy on 4 val samples, got {acc:.2f}"

@pytest.fixture(scope="session")
def iris_dm():
    dm = IrisDataModule(batch_size=32)
    dm.prepare_data()
    dm.setup()
    return dm
