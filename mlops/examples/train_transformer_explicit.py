"""Train a small Transformer with a manual PyTorch loop (no Lightning Trainer).

Reuses Transformer as a plain nn.Module and writes the loop by hand. The
bits Lightning was hiding: device placement, the next-token shift
(`tokens[..., :-1]` -> `tokens[..., 1:]`), gradient clipping, and a
no-grad validation pass averaging val_ce by hand.

This stays at fp32 for clarity. The Lightning version uses
`precision="bf16-mixed"`; the equivalent here would wrap the forward+loss
in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` plus a
`torch.amp.GradScaler` for the backward pass.

    uv run python mlops/examples/train_transformer_explicit.py
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from ft4.models.transformer import Transformer
from ft4.pipeline.stories_data_module import StoriesDataModule
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dm = StoriesDataModule(seq_len=256, batch_size=64, data_size="small")
dm.prepare_data()
dm.setup("fit")

model = Transformer(vocab_size=Ft4Tokenizer.vocab_size(), dim=128, depth=4).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

for epoch in range(2):
    model.train()
    for batch in dm.train_dataloader():
        tokens = batch["tokens"].to(device)
        x, y = tokens[..., :-1], tokens[..., 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.flatten(0, 1), y.flatten(), ignore_index=model.padding_idx
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    total_loss = num_batches = 0
    with torch.no_grad():
        for batch in dm.val_dataloader():
            tokens = batch["tokens"].to(device)
            x, y = tokens[..., :-1], tokens[..., 1:]
            logits = model(x)
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(), ignore_index=model.padding_idx
            )
            total_loss += loss.item()
            num_batches += 1
    print(f"epoch {epoch}  val_ce={total_loss/num_batches:.3f}")
