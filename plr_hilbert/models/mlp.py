
from typing import Tuple
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, return_features: bool = False):
        # x may be images [B,C,H,W]; flatten
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        feat = self.net(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits
