
from typing import Tuple
import torch

def batch_knn_pairs(X, k=2, topM=15, pseudo=None, pmax=None, min_conf=0.7, same_class_only=False):
    # X: [B,D] normalisé. On prend d'abord topM voisins puis on sous-échantillonne k par échantillon.
    with torch.no_grad():
        dist = torch.cdist(X, X, p=2)
        dist.fill_diagonal_(float("inf"))
        vals, idxM = torch.topk(dist, k=min(topM, X.size(0)-1), largest=False, dim=1)  # [B, topM]
        # Filtre optionnel (same class / pseudo-labels confiants)
        if pseudo is not None:
            same = (pseudo.unsqueeze(1) == pseudo[idxM])
            conf_ok = (pmax[idxM] > min_conf) if pmax is not None else torch.ones_like(same)
            mask = same if same_class_only else (same & conf_ok)
            # fallback: s'il n'y a pas assez de voisins valides, on garde les plus proches
            for b in range(X.size(0)):
                if mask[b].float().sum() < k:
                    mask[b, :] = True
            idxM = torch.where(mask, idxM, idxM)  # masque souple : on pourrait raffiner
        # échantillonne k
        if idxM.size(1) > k:
            choice = torch.randint(low=0, high=idxM.size(1), size=(X.size(0), k), device=X.device)
            idx = idxM.gather(1, choice)
        else:
            idx = idxM
        i_idx = torch.arange(X.size(0), device=X.device).unsqueeze(1).expand_as(idx).reshape(-1)
        j_idx = idx.reshape(-1)
    return i_idx, j_idx
