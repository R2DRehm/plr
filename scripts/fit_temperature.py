
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc Temperature Scaling for any saved run (val-fit, test-report)."""
import os, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from plr_hilbert.data import get_loaders
from plr_hilbert.models import MLP, SmallCNN
from plr_hilbert.losses.calibration import ece_score, brier_score, reliability_curve

class ModelWithTemperature(nn.Module):
    def __init__(self, model, init_T=1.0):
        super().__init__()
        self.model = model
        self.logT = nn.Parameter(torch.tensor([float(init_T)]).log())
    def forward(self, x):
        logits = self.model(x)
        T = self.logT.exp().clamp(min=1e-3, max=100.0)
        return logits / T

@torch.no_grad()
def eval_logits(model, loader, device, sigma=0.0):
    model.eval(); ce = nn.CrossEntropyLoss()
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    probs_all, labels_all = [], []
    for x, y in loader:
        if sigma>0: x = x + sigma * torch.randn_like(x)
        x, y = x.to(device), y.to(device)
        logits = model(x); loss = ce(logits, y)
        probs = F.softmax(logits, dim=1)
        tot_ce += float(loss.item()); tot_acc += float((logits.argmax(1)==y).float().sum().item()); n += y.size(0)
        probs_all.append(probs.cpu()); labels_all.append(y.cpu())
    probs_all = torch.cat(probs_all, 0); labels_all = torch.cat(labels_all, 0)
    return probs_all, labels_all, tot_ce / max(1, len(loader)), tot_acc / max(1, n)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["mnist","cifar10","cifar100","synthetic"])
    p.add_argument("--model", type=str, required=True, choices=["mlp","cnn_small","tinymlp"])
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--sigma", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if args.dataset=="synthetic":
        raise SystemExit("Use the synthetic script's built-in reporting.")
    loaders = get_loaders(args.dataset, data_dir="./data", batch_size=256, val_split=0.1, num_workers=2, seed=42)
    in_shape = loaders["meta"]["in_shape"]; num_classes = loaders["meta"]["num_classes"]
    if args.model=="mlp":
        in_dim = 1
        for s in in_shape: in_dim *= s
        base_model = MLP(in_dim, num_classes, hidden=256, dropout=0.0)
    else:
        base_model = SmallCNN(num_classes=num_classes, channels=in_shape[0], feat_dim=256, dropout=0.0)
    best = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device)
    base_model.load_state_dict(best["model"]); base_model.to(device)
    modelT = ModelWithTemperature(base_model).to(device)

    modelT.train()
    opt = torch.optim.LBFGS([modelT.logT], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        nll = 0.0
        for x, y in loaders["val"]:
            x, y = x.to(device), y.to(device); logits = modelT(x); nll = nll + ce(logits, y)
        nll = nll / max(1, len(loaders["val"]))
        nll.backward(); return nll
    opt.step(closure)

    modelT.eval()
    probs, labels, nll, acc = eval_logits(modelT, loaders["test"], device, sigma=args.sigma)
    metrics = {"acc": acc, "nll": nll, "ece": ece_score(probs, labels), "brier": brier_score(probs, labels), "T": float(modelT.logT.exp().item())}
    with open(os.path.join(args.run_dir, f"ts_metrics_sigma{args.sigma}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    mids, accs, confs = reliability_curve(probs, labels, n_bins=15)
    import matplotlib.pyplot as plt, numpy as np
    plt.figure(figsize=(6,5)); plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    m = ~np.isnan(accs); plt.plot(np.array(mids)[m], np.array(accs)[m], marker="o", label=f"TS σ={args.sigma}")
    plt.xlabel("Confidence (mean per bin)"); plt.ylabel("Empirical accuracy (per bin)")
    plt.title(f"Reliability — TS — σ={args.sigma} — T={metrics['T']:.2f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, f"ts_reliability_sigma{args.sigma}.png"), dpi=150); plt.close()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
