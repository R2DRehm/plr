
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic suite designed to favor PLR with strong logs."""
import os, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from plr_hilbert.losses.plr import plr_loss
from plr_hilbert.losses.calibration import ece_score, brier_score, reliability_curve
from plr_hilbert.utils.seed import set_seed

def make_dataset(n_train=6000, n_val=1000, n_test=2000, base_noise=0.12, seed=7):
    from sklearn.datasets import make_moons
    set_seed(seed)
    Xtr, ytr = make_moons(n_samples=n_train, noise=base_noise, random_state=seed)
    Xva, yva = make_moons(n_samples=n_val,   noise=base_noise, random_state=seed+1)
    Xte, yte = make_moons(n_samples=n_test,  noise=base_noise, random_state=seed+2)
    ytr = ytr.astype(np.int64)
    mask_local = np.abs(Xtr[:,1]) < 0.25
    flip_idx = np.where(mask_local & (np.random.rand(len(ytr)) < 0.25))[0]
    ytr[flip_idx] = 1 - ytr[flip_idx]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu)/sd; Xva = (Xva - mu)/sd; Xte = (Xte - mu)/sd
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    to_y = lambda a: torch.tensor(a, dtype=torch.long)
    return (to_t(Xtr), to_y(ytr)), (to_t(Xva), to_y(yva)), (to_t(Xte), to_y(yte))

class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, num_classes=2, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, return_features=False):
        h = F.relu(self.fc1(x))
        h = self.drop(F.relu(self.fc2(h)))
        z = self.head(h)
        if return_features: return z, h
        return z

def zscore(x):
    mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd

@torch.no_grad()
def evaluate(model, X, y, device, noise_sigma=0.0, n_bins=15):
    model.eval()
    Xe = X + noise_sigma * torch.randn_like(X) if noise_sigma>0 else X
    Xe, y = Xe.to(device), y.to(device)
    z = model(Xe); ce = F.cross_entropy(z, y, reduction="mean")
    p = F.softmax(z, dim=1); a = float((z.argmax(1)==y).float().mean().item())
    return {"acc":a, "nll": float(ce.item()), "ece": ece_score(p.cpu(), y.cpu(), n_bins=n_bins),
            "brier": brier_score(p.cpu(), y.cpu())}, p.cpu(), y.cpu()

def lambda_sched(epoch, warmup=5, ramp=15, lam_max=0.4):
    if epoch <= warmup: return 0.0
    t = min(1.0, (epoch - warmup) / float(max(1, ramp)))
    return lam_max * 0.5 * (1 - math.cos(math.pi * t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--plr_space", choices=["input","feature"], default="feature")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--ramp", type=int, default=25)
    ap.add_argument("--lam_max", type=float, default=0.4)
    ap.add_argument("--outdir", type=str, default="runs_synth")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    import pandas as pd
    all_rows = []
    for variant in ["base", "plr"]:
        for seed in args.seeds:
            set_seed(seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = make_dataset(seed=7+seed)
            model = TinyMLP().to(device)
            opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            tag = f"synth_{variant}_s{seed}"
            run_dir = os.path.join(args.outdir, tag); os.makedirs(run_dir, exist_ok=True)
            ds_tr = torch.utils.data.TensorDataset(Xtr, ytr)
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
            best_nll = float("inf"); best_path = os.path.join(run_dir, "best.pt")

            for ep in range(1, args.epochs+1):
                model.train()
                lam = lambda_sched(ep, warmup=args.warmup, ramp=args.ramp, lam_max=args.lam_max) if variant=="plr" else 0.0
                tot_ce=tot_plr=tot_acc=0.0; nb=0; run_active=0.0; run_ratio=0.0; calls=0
                for xb, yb in dl_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    zb, hb = model(xb, return_features=True)
                    loss_ce = F.cross_entropy(zb, yb); loss = loss_ce
                    if variant=="plr" and lam>0:
                        Xspace = F.normalize(hb.detach(), dim=1) if args.plr_space=="feature" else zscore(xb).detach()
                        lplr, info = plr_loss(zb, Xspace, k=args.k, tau=args.tau, reduction="mean", topM=15, smooth_range_t=10.0)
                        loss = loss + lam * lplr
                        tot_plr += float(lplr.item())
                        run_active += float(info.get("active_frac",0.0)); run_ratio += float(info.get("mean_ratio",0.0)); calls += 1
                    loss.backward(); opt.step()
                    tot_ce += float(loss_ce.item())
                    tot_acc += float((zb.argmax(1)==yb).float().mean().item())
                    nb += 1
                val_metrics, _, _ = evaluate(model, Xva, yva, device)
                if val_metrics["nll"] < best_nll:
                    best_nll = val_metrics["nll"]; torch.save({"model": model.state_dict()}, best_path)
                print(f"[{tag}] ep {ep:03d} | lam {lam:.3f} | CE {tot_ce/max(1,nb):.3f} PLR {tot_plr/max(1,nb):.3f} "
                      f"| act {(run_active/max(1,calls)):.3f} ratio {(run_ratio/max(1,calls)):.3f} "
                      f"| Val NLL {val_metrics['nll']:.3f} ECE {val_metrics['ece']:.3f}")

            if os.path.exists(best_path):
                state = torch.load(best_path, map_location=device); model.load_state_dict(state["model"])
            for sigma in [0.0, 0.25]:
                tm, p, y = evaluate(model, Xte, yte, device, noise_sigma=sigma)
                mids, accs, confs = reliability_curve(p, y, n_bins=15)
                import matplotlib.pyplot as plt, numpy as np
                plt.figure(figsize=(6,5)); plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
                m = ~np.isnan(accs); plt.plot(np.array(mids)[m], np.array(accs)[m], marker="o", label=f"σ={sigma}")
                plt.xlabel("Confidence (mean per bin)"); plt.ylabel("Empirical accuracy (per bin)")
                plt.title(f"Synthetic — {variant.upper()}"); plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(run_dir, f"reliability_sigma{sigma}.png"), dpi=150); plt.close()
                row = {"variant": variant, "seed": seed, "sigma": sigma, **tm}
                all_rows.append(row); print("TEST", row)

    os.makedirs("res", exist_ok=True)
    out_csv = os.path.join("res", "synthetic_plr_results.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == "__main__":
    main()
