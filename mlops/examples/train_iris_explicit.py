"""Train IrisClassifier with a manual PyTorch loop (no Lightning Trainer).

Reuses IrisClassifier — a LightningModule, which is also a torch.nn.Module —
and writes the train loop by hand. This is what trainer.fit() does for you:
the train/val phase split, the zero_grad/backward/step triple, train()/eval()
mode toggling, and the no-grad validation pass with manual metric averaging.

    uv run python mlops/examples/train_iris_explicit.py
"""
import torch
from torch.optim import AdamW
from ft4.models.iris import IrisClassifier, IrisDataModule

model = IrisClassifier()
dm = IrisDataModule(batch_size=32)
dm.prepare_data()
dm.setup()

optimizer = AdamW(model.parameters(), lr=3e-3)

for epoch in range(32):
    model.train()
    for x, y in dm.train_dataloader():
        optimizer.zero_grad()
        loss = model.criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"epoch {epoch:2d}  val_acc={correct/total:.3f}")
