import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import math

from ft4.models.mlp import MlpNet
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer

# Train this model: ft4 train path/to/model.py
# Then generate: ft4 generate path/to/model.py --prompt "One day"
# See mlops/README.md for details.
class Rnn(L.LightningModule):
    def __init__(
            self,
            dim: int = 128,
            expansion_factor: int = 4,
            depth: int = 8,
            
            lr: float = 3e-3,
            weight_decay: float = 1e-2,
            warmup_steps: int = 3000,
            vocab_size: int = Ft4Tokenizer.vocab_size(),
            padding_idx: int = 0):
        """
        Rnn predicts the next token according to a neural network, 
        conditioned on two things:
        1. The most recent token x_t
        2. A memory vector h, which summarizes everything in the text prior to x_t.
        
        At each pos, a single token is taken, and used to produce two outputs:
        1. the *next* token
        2. an updated memory vector h

        The simplest possible RNN is simply:
            h[t+1] = MLP(x_t + h_t)

        dim:                   Each token is embedded to a vector of `dim` dimensions
                               By far the most important hyperparameter:
                               A large dim is a powerful, but large, model
        vocab_size:            Size of vocabulary.
        padding_idx:           Token used to PAD
        
        `depth` and `expansion_factor` are not used by the starter code, 
        but you can choose to use them to determine the size and shape of nets you build.
        """
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        
        # Token ids -> vectors (embed), and vectors -> logits (unembed), tied.
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.unembed = nn.Linear(dim, vocab_size)
        self.unembed.weight = self.embed.weight # Tie weights
        self.scale_logits = 1.0 / math.sqrt(dim)

        # h0: learned initial memory, before any token has been read.
        self.initial_hidden_state = nn.Parameter(torch.randn(dim))

        # Initialize a neural net that will be used by `update_h`.
        # See `update_h` for requirements.
        # Hint: A single line of code may be enough.
        self.update_h_net = ... # TODO unit2.lab1
        raise NotImplementedError('Implement in unit2.lab1')

        # Initialize a neural net that will be used by `forward` (see there) 
        # to take each h vector and map it to a vector, of the same size, 
        # that predicts the *next* token:
        # The closer any token is to this vector, the higher the probability we give it.
        #
        # You can use any neural net you like to do this:
        # simply (a few lines of code), or you can be more fancy.
        # MlpNet (models/mlp.py), nn.Sequential, nn.Linear, nn.LayerNorm, nn.RMSNorm, etc. may be useful
        #
        self.prediction_head = ... # Shape: D --> D
        raise NotImplementedError('Implement in unit2.lab1') # TODO-LAB unit2.lab1

    def forward(self, tokens):
        """Next-token logits at every position.

        tokens:  (B, L) token ids
        returns: (B, L, V) logits
        Remember: logits[:, t] predicts tokens[:, t+1]

            input:        x0    x1    x2    x3
                           |     |     |     |
                    h0 --> h1 --> h2 --> h3 --> h4    h[t+1] summarizes x[0]...x[t]
                           |     |     |     |
                        prediction_head (same net at every position)
                           ↓     ↓     ↓     ↓
            predicts:     x1    x2    x3    x4

        The recurrence is serial, so h is computed in a loop.
        This makes RNNs big, slow, and hostile to GPUs.
        """
        B, L = tokens.shape
        x = self._token2vec(tokens) # shape: (B, L, D)
        h = self.initial_hidden_state.unsqueeze(0).expand(B, -1) # shape: (B, D)

        hs = []
        for i in range(L):
            # The recurrence is serial, so h is computed in a loop
            h = self.update_h(x[:, i, :], h)
            hs.append(h)

        # Training scores *all* L predictions (see _prediction_loss), so we can learn from our predictions at each point of the sequence.
        # Generation would need only the final h.
        # For efficiency, we stack all the h and run them through prediction_head in a single shot
        h_seq = torch.stack(hs, dim=1) # shape: (B, L, D)
        
        # prediction_head turns *each* h into a single vector;
        # the closer any token in our vocab is to this vector, the greater its logit will be
        predictions = self.prediction_head(h_seq) # shape: (B, L, D)
        # unembed uses this vector to assign a logit to each token in our vocab
        # for stability, we scale them down by sqrt(dim)
        logits = self.unembed(predictions) * self.scale_logits
        return logits

    def update_h(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: A single token in vector form. Shape: (B, D).
        h: Current hidden state. Shape: (B, D).
        
        returns: Updated hidden state. Shape: (B, D).
        """
        # Use a simple neural net (i.e. the update_h_net net you created in __init__) 
        # to consume x and h and output next_h.
        #
        # How will you feed both x and h to the net?
        # You might concatenate them, or even just add them together.
        
        next_h = ... # TODO-LAB unit2.lab1
        raise NotImplementedError('Implement in unit2.lab1')
        return next_h
    
    def _token2vec(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: A batch of sequences of token ids (integers). Shape: (B, L).
        returns: The batch, with each token transformed to its vector representation. Shape: (B, L, D).
        """
        x = self.embed(tokens)
        assert x.shape == tokens.shape + (self.dim,)
        return x
    
    def _prediction_loss(self, batch) -> torch.Tensor:
        """Cross-entropy of predicting tokens[:, t+1], averaged over all t."""
        tokens = batch['tokens']  # (B, L+1)
        x = tokens[..., :-1]      # (B, L)  x[b, t]: token at time t
        y = tokens[..., 1:]       # (B, L)  y[b, t]: token at time t+1 (target)
        logits = self(x)          # (B, L, V)

        # F.cross_entropy requires logits shape (N, V) and targets shape (N),
        # so we flatten B,L -> B*L
        return F.cross_entropy(
            input=logits.flatten(start_dim=0, end_dim=1),  # (B*L, V)
            target=y.flatten(),                            # (B*L,)
            ignore_index=self.padding_idx)                 # ignore_index masks padded targets (this pipeline shouldn't have any).

    def training_step(self, batch):
        loss = self._prediction_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        loss = self._prediction_loss(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):  # type:ignore
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)  # type:ignore
        # TODO Add cosine annealing scheduler
        return {'optimizer': optimizer}
