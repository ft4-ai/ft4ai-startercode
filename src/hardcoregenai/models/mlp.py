import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self, dim: int, expansion_factor: int=4):
        super().__init__()
        hdim = dim * expansion_factor
        dropout = 0.1

        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hdim),
            #nn.GELU(),
            ReLUSquared(), # ReLUSquared beats GELU by about 0.06 CE in my simple experiments
            nn.Dropout(dropout),
            nn.Linear(hdim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.layers(x) # Add in x as a residual
    
class MlpNet(nn.Module):
    def __init__(self, depth: int, dim: int, expansion_factor: int=4):
        super().__init__()
        self.layers = nn.ModuleList([MlpBlock(dim, expansion_factor) for _ in range(depth)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

class ReLUSquared(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x) ** 2
