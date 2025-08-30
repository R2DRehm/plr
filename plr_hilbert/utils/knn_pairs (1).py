
from typing import Tuple, Optional
import torch

def batch_knn_pairs(X: torch.Tensor, k: int = 2, topM: int = 15,
                    pseudo: Optional[torch.Tensor] = None,
                    pmax: Optional[torch.Tensor] = None,
                    min_conf: float = 0.7,
                    same_class_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (i_idx, j_idx) of k neighbors for each sample in batch, with topM preselection.
    Optionally filter by pseudo-label agreement and confidence.

    Note: If too few masked neighbors for a row, we fallback to the closest ones.
    """
    assert X.dim() == 2
    B = X.size(0)
    with torch.no_grad():
        dist = torch.cdist(X, X, p=2)
        dist.fill_diagonal_(float("inf"))
        M = min(topM, max(1, B - 1))
        valsM, idxM = torch.topk(dist, k=M, largest=False, dim=1)  # [B,M]

        if pseudo is not None:
            same = (pseudo.unsqueeze(1) == pseudo[idxM])
            conf_ok = (pmax[idxM] > min_conf) if pmax is not None else torch.ones_like(same, dtype=torch.bool)
            mask = same if same_class_only else (same & conf_ok)
            for b in range(B):
                if mask[b].sum() < max(1, k):
                    mask[b] = True

        if M > k:
            choice = torch.randint(low=0, high=M, size=(B, k), device=X.device)
            nbrs = idxM.gather(1, choice)
        else:
            nbrs = idxM[:, :k]

        i_idx = torch.arange(B, device=X.device).unsqueeze(1).expand_as(nbrs).reshape(-1)
        j_idx = nbrs.reshape(-1)
    return i_idx, j_idx
