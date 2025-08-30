
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run baseline vs PLR on MNIST/CIFAR with strong logging and safe defaults."""
import os, math, argparse, csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from plr_hilbert.data import get_loaders
from plr_hilbert.models import MLP, SmallCNN
from plr_hilbert.losses.plr import plr_loss
from plr_hilbert.losses.calibration import ece_score, brier_score, reliability_curve
from plr_hilbert.utils.seed import set_seed

def build_model(dataset: str, model_name: str, in_shape, num_classes: int):
    model_name = model_name.lower()
    if model_name == "mlp":
        in_dim = 1
        for s in in_shape: in_dim *= s
        return MLP(in_dim=in_dim, num_classes=num_classes, hidden=256, dropout=0.1)
    elif model_name in ["cnn_small","cnn","smallcnn"]:
        channels = in_shape[0]
        return SmallCNN(num_classes=num_classes, channels=channels, feat_dim=256, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean().item())

@torch.no_grad()
def evaluate(model, loader, device, noise_sigma=0.0, n_bins=15):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    probs_all, labels_all = [], []
    for x, y in loader:
        if noise_sigma and noise_sigma > 0: x = x + noise_sigma * torch.randn_like(x)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_ce = ce(logits, y)
        probs = F.softmax(logits, dim=1)
        tot_ce += float(loss_ce.item())
        tot_acc += float((logits.argmax(1) == y).float().sum().item())
        n += y.size(0)
        probs_all.append(probs.cpu())
        labels_all.append(y.cpu())
    probs_all = torch.cat(probs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return {
        "acc": tot_acc / max(1, n),
        "nll": tot_ce / max(1, len(loader)),
        "ece": ece_score(probs_all, labels_all, n_bins=n_bins),
        "brier": brier_score(probs_all, labels_all),
    }, probs_all, labels_all

def lambda_sched(epoch, warmup=5, ramp=15, lam_max=0.4):
    if epoch <= warmup: return 0.0
    t = min(1.0, (epoch - warmup) / float(max(1, ramp)))
    return lam_max * 0.5 * (1 - math.cos(math.pi * t))

def zscore_flat(x):
    flat = x.view(x.size(0), -1)
    mu = flat.mean(dim=1, keepdim=True)
    sd = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (flat - mu) / sd

def train_epoch(model, loader, optimizer, device, use_plr, plr_space, tau, k, lam,
                detach_features=True):
    model.train()
    ce = nn.CrossEntropyLoss()
    tot_ce = tot_plr = tot_acc = 0.0
    n_batches = 0
    running_active = 0.0
    running_ratio  = 0.0
    num_plr_calls  = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if plr_space == "feature":
            logits, feat = model(x, return_features=True)
            Xspace = F.normalize(feat.detach() if detach_features else feat, dim=1)
        else:
            logits = model(x)
            Xspace = zscore_flat(x).detach()
        loss_ce = ce(logits, y)
        loss = loss_ce
        if use_plr and lam > 0:
            l_plr, info = plr_loss(logits=logits, X_space=Xspace, k=k, tau=tau,
                                   reduction="mean", topM=15, smooth_range_t=10.0)
            loss = loss + lam * l_plr
            tot_plr += float(l_plr.item())
            running_active += float(info.get("active_frac", 0.0))
            running_ratio  += float(info.get("mean_ratio", 0.0))
            num_plr_calls  += 1
        loss.backward()
        optimizer.step()
        tot_ce  += float(loss_ce.item())
        tot_acc += acc(logits, y)
        n_batches += 1
    return {
        "train_ce":  tot_ce  / max(1, n_batches),
        "train_plr": tot_plr / max(1, n_batches),
        "train_acc": tot_acc / max(1, n_batches),
        "active_frac": (running_active / max(1, num_plr_calls)),
        "mean_ratio":  (running_ratio  / max(1, num_plr_calls)),
    }

def run_once(dataset, model_name, seed, outdir,
             epochs=50, batch_size=128, lr=1e-3, weight_decay=1e-4,
             use_cuda=True,
             use_plr=False, plr_space="feature",
             tau=0.40, k=2, warmup=10, ramp=20, lam_max=0.4):
    set_seed(seed)
    loaders = get_loaders(dataset, data_dir="./data", batch_size=batch_size, val_split=0.1, num_workers=2, seed=seed)
    in_shape = loaders["meta"]["in_shape"]; num_classes = loaders["meta"]["num_classes"]
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = build_model(dataset, model_name, in_shape, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tag = f"{dataset}_{model_name}_{'plr' if use_plr else 'base'}_s{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(outdir, tag); os.makedirs(run_dir, exist_ok=True)
    best_nll = float("inf"); best_path = os.path.join(run_dir, "best.pt")
    for ep in range(1, epochs+1):
        lam = lambda_sched(ep, warmup=warmup, ramp=ramp, lam_max=lam_max) if use_plr else 0.0
        stats = train_epoch(model, loaders["train"], optimizer, device, use_plr, plr_space, tau, k, lam, detach_features=True)
        val_metrics, _, _ = evaluate(model, loaders["val"], device)
        row = {"epoch": ep, **stats, **{f"val_{k}": v for k, v in val_metrics.items()}, "lam": lam}
        with open(os.path.join(run_dir, "metrics_epoch.csv"), "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0: w.writeheader()
            w.writerow(row)
        print(f"[{tag}] ep {ep:03d} | lam {lam:.3f} | CE {stats['train_ce']:.3f} PLR {stats['train_plr']:.3f} "
              f"| act {stats['active_frac']:.3f} ratio {stats['mean_ratio']:.3f} "
              f"| Val Acc {val_metrics['acc']:.3f} NLL {val_metrics['nll']:.3f} ECE {val_metrics['ece']:.3f}")
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]; torch.save({"model": model.state_dict()}, best_path)
    # Test (clean + noisy)
    all_rows = []
    for sigma in [0.0, 0.25]:
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state["model"])
        test_metrics, probs, labels = evaluate(model, loaders["test"], device, noise_sigma=sigma, n_bins=15)
        mids, accs, confs = reliability_curve(probs, labels, n_bins=15)
        import matplotlib.pyplot as plt, numpy as np
        plt.figure(figsize=(6,5)); plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
        m = ~np.isnan(accs); plt.plot(np.array(mids)[m], np.array(accs)[m], marker="o", label=f"σ={sigma}")
        plt.xlabel("Confidence (mean per bin)"); plt.ylabel("Empirical accuracy (per bin)")
        plt.title(f"Reliability — {dataset.upper()} — {'PLR' if use_plr else 'Baseline'}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"reliability_sigma{sigma}.png"), dpi=150); plt.close()
        all_rows.append({"dataset": dataset, "model": model_name, "seed": seed, "variant": ("plr" if use_plr else "base"),
                         "sigma": sigma, **test_metrics})
    with open(os.path.join(run_dir, "metrics_test.csv"), "w", newline="") as f:
        keys = list(all_rows[0].keys()); w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in all_rows: w.writerow(r)
    return run_dir, all_rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["mnist","cifar10","cifar100"], default="mnist")
    p.add_argument("--model", type=str, choices=["mlp","cnn_small"], default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--plr_space", type=str, default=None, choices=["input","feature"])
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--ramp", type=int, default=None)
    p.add_argument("--lam_max", type=float, default=None)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--aggregate_to", type=str, default="res/bench_results.csv")
    args = p.parse_args()

    if args.model is None: args.model = "mlp" if args.dataset=="mnist" else "cnn_small"
    if args.plr_space is None: args.plr_space = "input" if args.dataset=="mnist" else "feature"
    if args.tau is None: args.tau = 0.5 if args.dataset=="mnist" else 0.4
    if args.warmup is None: args.warmup = 5 if args.dataset=="mnist" else 10
    if args.ramp is None: args.ramp = 15 if args.dataset=="mnist" else 20
    if args.lam_max is None: args.lam_max = 0.3 if args.dataset=="mnist" else 0.4

    os.makedirs(os.path.dirname(args.aggregate_to), exist_ok=True)

    all_rows = []
    for variant in ["base","plr"]:
        use_plr = (variant=="plr")
        for seed in args.seeds:
            run_dir, rows = run_once(
                dataset=args.dataset, model_name=args.model, seed=seed, outdir=args.outdir,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                use_cuda=args.use_cuda, use_plr=use_plr, plr_space=args.plr_space, tau=args.tau, k=args.k,
                warmup=args.warmup, ramp=args.ramp, lam_max=args.lam_max
            )
            all_rows.extend(rows)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(args.aggregate_to, index=False)
    print(df.groupby(["dataset","variant","sigma"])[["acc","nll","ece","brier"]].mean().round(4))

if __name__ == "__main__":
    main()
