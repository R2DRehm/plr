
from typing import Tuple
import torch

def batch_knn_pairs(X: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (i_idx, j_idx) of k nearest neighbors for each sample in batch.
    X: (B, D) tensor. Uses Euclidean distance within batch.
    Returns: i_idx, j_idx of shape (B*k,).
    """
    assert X.dim() == 2, "X should be [B, D]"
    B = X.shape[0]
    # Compute pairwise distances
    with torch.no_grad():
        # cdist yields [B,B]
        dist = torch.cdist(X, X, p=2)
        dist.fill_diagonal_(float("inf"))
        # topk smallest distances
        vals, idx = torch.topk(dist, k=k, largest=False, dim=1)  # idx: [B,k]
        i_idx = torch.arange(B, device=X.device).unsqueeze(1).repeat(1, k).reshape(-1)
        j_idx = idx.reshape(-1)
    return i_idx, j_idx
