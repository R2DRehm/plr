
from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """L1 ECE with equal-width bins over confidence of predicted class."""
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    bins = torch.linspace(0., 1., n_bins + 1, device=probs.device)
    ece = torch.tensor(0., device=probs.device)
    N = probs.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi + 1e-12)
        if m.any():
            bin_acc = acc[m].mean()
            bin_conf = conf[m].mean()
            ece = ece + (m.float().mean() * (bin_acc - bin_conf).abs())
    return float(ece.item())

@torch.no_grad()
def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.zeros_like(probs).scatter_(1, labels.view(-1,1), 1.0)
    return float(((probs - one_hot) ** 2).sum(dim=1).mean().item())

@torch.no_grad()
def reliability_curve(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    bins = torch.linspace(0., 1., n_bins + 1, device=probs.device)
    mids, accs, confs = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi + 1e-12)
        mids.append(float((lo + hi).item()/2))
        if m.any():
            accs.append(float(acc[m].mean().item()))
            confs.append(float(conf[m].mean().item()))
        else:
            accs.append(float("nan"))
            confs.append(float("nan"))
    return np.array(mids), np.array(accs), np.array(confs)

def plot_reliability(mids, accs, confs, title: str, out_path: str):
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    m = ~np.isnan(accs)
    plt.plot(mids[m], accs[m], marker="o", label="Model")
    plt.xlabel("Confidence (mean per bin)")
    plt.ylabel("Empirical accuracy (per bin)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
