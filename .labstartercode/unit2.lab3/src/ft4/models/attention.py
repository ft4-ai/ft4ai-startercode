import math

import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int=8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.attn_scalar = 1/math.sqrt(dim//n_heads)
        
        self.Q = nn.Linear(dim,  dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        # O is used for multi-headed attention, discussed at the end of Unit 2 Lab 3
        # You can ignore it until then
        self.O = nn.Linear(dim, dim)

        # LayerNorm and Dropout aren't fundamental to attention
        # But they improve results and are in practice always used
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, L, D)
        return: shape (B, L, D)
        """
        return x + self.attend(self.layer_norm(x))
        
    def attend(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, L, D)
        return: shape (B, L, D)
        """
        B = x.shape[0]
        L = x.shape[1]
        assert x.shape == (B, L, self.dim)
        head_dim = self.dim//self.n_heads
        assert head_dim * self.n_heads == self.dim
        
        q = self.Q(x) # shape: (B, L, D)
        k = self.K(x)
        v = self.V(x)

        # Determine the weighted_vals for the context
        # This should take about a dozen lines
        weighted_vals = ... # TODO-LAB unit2.lab3
        raise NotImplementedError('Implement in unit2.lab3')

        outs = self.O(weighted_vals)
        outs = self.dropout(outs)
        return outs
