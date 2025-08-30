
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate results from runs/* and produce CSVs + LaTeX table (σ=0.0)."""
import os, glob, argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", type=str, default="runs/*/metrics_test.csv")
    p.add_argument("--out_csv", type=str, default="res/bench_aggregate.csv")
    p.add_argument("--latex_out", type=str, default="res/table_plr.tex")
    args = p.parse_args()

    paths = glob.glob(args.runs_glob)
    if not paths: print("No runs found:", args.runs_glob); return
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False); print("Saved", args.out_csv)
    g = df.groupby(["dataset","variant","sigma"])[["acc","nll","ece","brier"]].agg(["mean","std"]).round(4)
    print(g)
    d0 = df[df["sigma"]==0.0].groupby(["dataset","variant"])[["acc","nll","ece","brier"]].mean().round(4).unstack("variant")
    with open(args.latex_out, "w") as f: f.write(d0.to_latex())
    print("Saved LaTeX to", args.latex_out)

if __name__ == "__main__":
    main()
