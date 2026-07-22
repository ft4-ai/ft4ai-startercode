# Remember: This is *your* code base. Experiment, modify, explore!
#
# Half of AI coding is tracking the tensor shapes correctly:
# It's often helpful to modify the code to print the shape
# and some sample data of important tensors.


import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L

from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

# There's nothing to train here, but you can still generate gibberish:
#   ft4 generate path/to/model.py --prompt "One day"
class UniformModel(L.LightningModule):
    """
    The UniformModel is the simplest possible model that fits the pipeline.

    It predicts the same probability for every candidate token (!),
    and hence has *no* predictive value. But it's a great way to get started,
    and smoke tests the pipeline end-to-end.

    UniformModel has no learnable parameters, so:
      - automatic_optimization is off,
      - training_step is a no-op,
      - configure_optimizers returns None.
    validation_step still reports cross-entropy, giving a baseline any real model must beat.
    """
    def __init__(self, vocab_size: int = Ft4Tokenizer.vocab_size()):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.automatic_optimization = False

    def forward(self, tokens: Tensor) -> Tensor:
        """
        tokens:  A batch of sequences of token IDs
                 Shape: (B,L) int.
        returns: Prediction logits: For each position in the sequence,
                 a distribution over the vocab for the *next* token.
                 Shape: (B,L,V) float.
        where:
         B = batch size
         L = tokens per sequence
         V = vocab size
        """
        B, L = tokens.shape
        V = self.vocab_size
        # Determine the logits:
        # The UniformModel trivially predicts an identical logit for *every* candidate in the vocab.
        logits = ... # TODO-LAB unit1.lab1
        raise NotImplementedError("Implement in unit1.lab1")
        return logits

    def training_step(self, batch, batch_idx):
        # UniformModel has no parameters; there is nothing to learn from this batch.
        return None

    def validation_step(self, batch):
        # For a uniform distribution over V tokens, this loss equals log(V) --
        # the entropy floor that any real language model must beat.
        tokens = batch['tokens']
        x = tokens[..., :-1]
        y = tokens[..., 1:]
        logits = self(x)
        logits_ = logits.flatten(start_dim=0, end_dim=1)
        y_ = y.flatten()
        loss = F.cross_entropy(input=logits_, target=y_, ignore_index=Ft4Tokenizer.RES_PAD)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Nothing to optimize.
        return None
