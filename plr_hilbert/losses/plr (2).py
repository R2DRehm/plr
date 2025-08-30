
# Drop-in replacement for plr_hilbert/losses/plr.py
# - Robust distance scaling (median-of-pairs)
# - Optional smooth range (log-sum-exp) for stable gradients
# - Active fraction & mean ratio diagnostics
# - Top-M neighbor preselection
from typing import Tuple, Optional
import torch

def _srange_t(v: torch.Tensor, t: float = 10.0) -> torch.Tensor:
    # smooth range via log-sum-exp; t in [8,12] works well
    smax = torch.logsumexp( v * t, dim=-1) / t
    smin = -torch.logsumexp(-v * t, dim=-1) / t
    return smax - smin

def plr_loss(
    logits: torch.Tensor,
    X_space: torch.Tensor,
    k: int = 2,
    tau: float = 0.30,
    reduction: str = "mean",
    eps: float = 1e-8,
    topM: int = 15,
    smooth_range_t: Optional[float] = None,  # e.g. 10.0; None = hard range
) -> Tuple[torch.Tensor, dict]:
    """
    Projective-Lipschitz Regularization (PLR) with diagnostics.
    Penalizes the Hilbert-projective variation of logits between within-batch k-NN pairs.

    Args:
        logits: [B, K] tensor of logits.
        X_space: [B, D] features where distances are computed.
        k: neighbors per sample (2-4 recommended).
        tau: threshold; only penalize variations exceeding tau.
        reduction: 'mean' (default) or 'sum'.
        topM: preselect M nearest neighbors before sampling k per anchor (robust).
        smooth_range_t: if set, use smooth range (log-sum-exp) with temperature t.
    Returns:
        (loss, info_dict)
    """
    assert logits.dim() == 2 and X_space.dim() == 2 and logits.size(0) == X_space.size(0)
    B, K = logits.shape
    if B <= 2:
        z = logits.new_zeros(())
        return z, {"active_frac": 0.0, "mean_ratio": 0.0}

    with torch.no_grad():
        # Pairwise distances
        dist = torch.cdist(X_space, X_space, p=2)
        dist.fill_diagonal_(float("inf"))
        # Preselect topM nearest neighbors
        M = min(topM, max(1, B - 1))
        valsM, idxM = torch.topk(dist, k=M, largest=False, dim=1)  # [B,M]
        # Choose k from topM uniformly (avoids always picking exact same neighbors)
        if M > k:
            choice = torch.randint(low=0, high=M, size=(B, k), device=logits.device)
            nbrs = idxM.gather(1, choice)
        else:
            nbrs = idxM[:, :k]
        i_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand_as(nbrs).reshape(-1)
        j_idx = nbrs.reshape(-1)

    dz = logits[i_idx] - logits[j_idx]            # [P, K]
    if smooth_range_t is None:
        rng = dz.max(dim=1).values - dz.min(dim=1).values  # [P]
    else:
        rng = _srange_t(dz, t=float(smooth_range_t))       # [P]

    dX = X_space[i_idx] - X_space[j_idx]          # [P, D]
    dx = dX.norm(dim=1) + eps                     # [P]
    # Robust per-batch scaling: use median distance as scale 1.0
    scale = dx.detach().median()
    ratios = rng / (dx / (scale + eps))

    over = ratios - tau
    active = over > 0
    if reduction == "sum":
        loss = torch.relu(over).sum() / max(1, logits.shape[0])
    else:
        loss = torch.relu(over).mean()

    info = {
        "active_frac": float(active.float().mean().item()),
        "mean_ratio": float(ratios.mean().item()),
        "tau": float(tau),
        "scale_median_dx": float(scale.item()),
    }
    return loss, info
