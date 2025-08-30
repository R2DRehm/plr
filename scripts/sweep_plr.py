
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid sweep for PLR hyperparams; picks best by validation NLL."""
import os, argparse, itertools
import pandas as pd
from run_plr_bench import run_once

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["mnist","cifar10","cifar100"], default="mnist")
    p.add_argument("--model", type=str, choices=["mlp","cnn_small"], default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--lam_max", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    p.add_argument("--tau", type=float, nargs="+", default=[0.35, 0.4, 0.45])
    p.add_argument("--k", type=int, nargs="+", default=[2, 3])
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--out", type=str, default="res/sweep.csv")
    args = p.parse_args()

    if args.model is None: args.model = "mlp" if args.dataset=="mnist" else "cnn_small"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    # baseline
    for seed in args.seeds:
        _, r = run_once(args.dataset, args.model, seed, args.outdir, epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, weight_decay=args.weight_decay, use_cuda=args.use_cuda,
                        use_plr=False, plr_space=("input" if args.dataset=="mnist" else "feature"),
                        tau=0.0, k=2, warmup=5, ramp=15, lam_max=0.0)
        rows += r

    # grid
    for lam in args.lam_max:
        for tau in args.tau:
            for k in args.k:
                for seed in args.seeds:
                    _, r = run_once(args.dataset, args.model, seed, args.outdir, epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, weight_decay=args.weight_decay, use_cuda=args.use_cuda,
                                    use_plr=True, plr_space=("input" if args.dataset=="mnist" else "feature"),
                                    tau=tau, k=k, warmup=(5 if args.dataset=='mnist' else 10), ramp=(15 if args.dataset=='mnist' else 20),
                                    lam_max=lam)
                    for rr in r:
                        rr["lam_max"] = lam; rr["tau"] = tau; rr["k"] = k
                    rows += r
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print("Saved sweep to", args.out)
    print(df.groupby(["dataset","variant","sigma"])[["acc","nll","ece","brier"]].mean().round(4))

if __name__ == "__main__":
    main()
