
from typing import Tuple, Optional
import torch
from ..utils.knn_pairs import batch_knn_pairs

def srange_t(v, t=10.0):
    smax = (v * t).logsumexp(dim=-1) / t
    smin = (-v * t).logsumexp(dim=-1) / t
    return smax - smin

def plr_loss(logits, X_space, k=2, tau=0.30, reduction="mean", eps=1e-8, topM=15):
    i_idx, j_idx = batch_knn_pairs(X_space, k=min(k, topM), topM=topM)  # vois plus bas
    dz = logits[i_idx] - logits[j_idx]
    rng = srange_t(dz, t=10.0)                           # <— lissé
    dx = (X_space[i_idx] - X_space[j_idx]).norm(dim=1) + eps
    scale = dx.detach().median()                         # <— échelle robuste
    ratios = rng / (dx / scale)
    over = ratios - tau
    loss = torch.relu(over).mean() if reduction == "mean" else torch.relu(over).sum() / logits.size(0)
    info = {"active_frac": float((over > 0).float().mean().item()),
            "mean_ratio": float(ratios.mean().item())}
    return loss, info
