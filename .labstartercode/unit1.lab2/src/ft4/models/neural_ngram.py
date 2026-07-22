import math

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW
import lightning as L

from ft4.models.mlp import MlpNet
from ft4.pipeline.bpe_tokenizer import Ft4Tokenizer


# Train this model: ft4 train path/to/model.py
# Then generate:    ft4 generate path/to/model.py
# See mlops/README.md.
class NeuralNgram(L.LightningModule):
    def __init__(
            self,
            n: int = 3,
            dim: int = 128,
            depth: int = 8,
            expansion_factor: int = 4,
            lr: float = 1e-3,
            warmup_steps: int=3000,
            vocab_size: int = Ft4Tokenizer.vocab_size(),
            padding_idx: int = 0):
        """
        NeuralNgram predicts the next token according to a neural network,
        conditioned on the previous w tokens, where w = n - 1.

        At each pos, w tokens (including the current token) are taken as a lumped input, 
        and used to predict the *next* token.

        n:                     The context window w used for prediction is n - 1.
        dim:                   Each token is embedded to a vector of `dim` dimensions
                               By far the most important hyperparameter after n:
                               A large dim is a powerful, but large, model
        vocab_size:            Size of vocabulary.
        padding_idx:           Token used to PAD
        
        `depth` and `expansion_factor` are not used by the starter code, 
        but you can choose to use them to determine the size and shape of nets you build.
        """
        super().__init__()
        self.save_hyperparameters()
        assert n > 0
        self.n = n
        self.w = n - 1
        """w (window): The number of preceding tokens we condition on to predict the next one"""
        self.dim = dim
        self.depth = depth
        self.padding_idx = padding_idx

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        # TODO Doc unembed
        self.unembed = nn.Linear(dim, vocab_size, bias=False)
        self.unembed.weight = self.embed.weight # Tie weights
        
        # Initialize a neural net that transforms batches of embedded-ngrams [shape (..., N*D)]
        # to predicted embeddings of the *next* token [shape (...., D)]
        # 
        # You can use any neural net you like to do this:
        # simply (a few lines of code), or you can be more fancy.
        # MlpNet (models/mlp.py), nn.Sequential, nn.Linear, nn.LayerNorm, nn.RMSNorm, etc. may be useful
        #
        # TODO-LAB unit1.lab2
        self.net = ... 
        raise NotImplementedError('TODO-LAB unit.lab2')

    def forward(self, tokens: Tensor) -> Tensor:
        """
        tokens:  A batch of sequences of token IDs
                 Shape: (B,L) int.
        returns: Prediction logits:
                 For each position in the sequence, a distribution over the vocab for the *next* token.
                 Shape: (B,L,V) float.
        where:
         B = batch size
         L = tokens per sequence
         V = vocab size
        """
        assert tokens.dtype == torch.long
        # Map each token id to a `dim`-dimensional vector
        embedded_tokens = self.embed(tokens) # (B, L, D) where D is self.dim
        # Find all ngrams in the sequence
        ngrams = self._make_ngrams(embedded_tokens) # (B, L, N*D) where N is self.n
        # And map each ngram to logits over the vocab
        logits = self._predict_next(ngrams) # (B, L, V) where V is self.vocab_size
        return logits

    def _predict_next(self, ngrams: torch.Tensor) -> Tensor:
        """
        ngrams:    Vectorized ngrams, one per pos in the seq
                   Shape: (B, L, N*D) float
        returns:   Prediction logits:
                   For each position in the sequence, a distribution over the vocab for the *next* token.
                   Shape: (B, L, V) float
        """
        # First, map each N*D ngram to a single D-dimensional vector
        # You might do this using the `self.net` you built in `__init__()`
        # TODO-LAB unit1.lab2
        
        # Then, unembed each D vector to a V-dimensional to form logits over the vocab
        # The unembedding is usually a simple linear transformation using the embedding table weights
        logits = None # TODO-LAB unit1.lab2
        raise NotImplementedError("Implement _predict_next in unit1.lab2")
        return logits

    def _make_ngrams(self, embedded_tokens: Tensor) -> Tensor:
        """
        Finds all ngrams in embedded_tokens.
        To allow a full ngram for pos 0, seqs are padded left with n-1 padding tokens.
        
        Example:
          batch_size = 1, seq_len = 3, n = 3, dim = 2
          embedded_tokens = [ [1,2], [3,4], [5,6] ]
          # Assume padding_idx = 0 embeds to [0,0]

          returns: [ [0,0,0,0,1,2], 
                     [0,0,1,2,3,4], 
                     [1,2,3,4,5,6] ]
        
        embedded_tokens: Shape: (B, L, D)
        returns:         Shape: (B, L, N*D)
        """
        if self.n <= 1:
            return embedded_tokens
        
        B, L, D = embedded_tokens.shape

        # Left pad (to allow a full ngram at pos 0)        
        pad_vec = self.embed.weight[self.padding_idx].detach()
        left_pad = pad_vec.view(1, 1, -1).expand(B, self.w, D)
        padded = torch.cat([left_pad, embedded_tokens], dim=1)  # (B, L+w, D)

        # Take sliding windows of length n over dim=1
        windows = padded.unfold(dimension=1, size=self.n, step=1) # (B, L, n, D)
        ngrams = windows.reshape(B, L, self.n * D) # (B, L, n*D)
        return ngrams
    
    def training_step(self, batch):
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def _loss(self, batch):
        tokens = batch['tokens'] # shape: (B, L+1)
        x = tokens[..., :-1]     # shape: (B, L) 
                                 #    x[b][i] is the token id at pos i (in batch b)
        y = tokens[..., 1:]      # shape: (B, L) 
                                 #    y[b][i] is the token id at pos i+1 (and thus the target for pos i)
        logits = self(x)         # shape: (B, L, V)
        
        logits_ = logits.flatten(start_dim=0, end_dim=1) # shape: (B*L, V)
        y_ = y.flatten()                                 # shape: (B*L,)
        loss = F.cross_entropy(input=logits_, target=y_, ignore_index=self.padding_idx)

        return loss

    def configure_optimizers(self): # type:ignore
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return {'optimizer': optimizer}
    
    def __str__(self):
        return f'NeuralNgram(n={self.n} dim={self.dim} depth={self.depth})'
