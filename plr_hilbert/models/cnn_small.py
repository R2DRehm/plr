
import torch.nn as nn
import torch

class SmallCNN(nn.Module):
    """A lightweight CNN for 32x32 images (e.g., CIFAR-10)."""
    def __init__(self, num_classes: int = 10, channels: int = 3, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, feat_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.features(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits
