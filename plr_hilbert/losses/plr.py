
from typing import Tuple, Optional
import torch
from ..utils.knn_pairs import batch_knn_pairs

def plr_loss(logits: torch.Tensor,
             X_space: torch.Tensor,
             k: int = 2,
             tau: float = 0.30,
             reduction: str = "mean",
             eps: float = 1e-8) -> Tuple[torch.Tensor, dict]:
    """Projective-Lipschitz Regularization (PLR).
    Penalizes the Hilbert-projective variation of logits between within-batch k-NN pairs.

    Args:
        logits: [B, K] tensor of logits.
        X_space: [B, D] features where distances are computed (inputs or penultimate features).
        k: number of neighbors per sample (2-4 recommended).
        tau: threshold; only penalize variations exceeding tau.
        reduction: 'mean' (default) or 'sum'.
    Returns:
        (loss, info_dict) where loss is a scalar tensor and info has diagnostics
    """
    assert logits.dim() == 2 and X_space.dim() == 2 and logits.size(0) == X_space.size(0)
    B, K = logits.shape
    if B <= 2:
        return logits.new_zeros(()), {"active_frac": 0.0, "mean_ratio": 0.0}

    i_idx, j_idx = batch_knn_pairs(X_space, k=k)  # shapes [P]
    dz = logits[i_idx] - logits[j_idx]            # [P, K]
    # Hilbert projective distance on softmax outputs equals range of dz
    rng = dz.max(dim=1).values - dz.min(dim=1).values  # [P]
    dx = torch.norm(X_space[i_idx] - X_space[j_idx], dim=1) + eps  # [P]
    scale = dx.detach().median()         # robuste
    ratios = rng / (dx / scale)
    over = ratios - tau
    active = over > 0
    if reduction == "sum":
        loss = torch.relu(over).sum() / max(1, logits.shape[0])
    else:
        loss = torch.relu(over).mean()

    info = {
        "active_frac": float(active.float().mean().item()),
        "mean_ratio": float(ratios.mean().item()),
    }
    return loss, info
