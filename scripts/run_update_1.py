#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_update_1.py — CE & PLR sur MNIST et CIFAR-10
- Utilise --model (et non --model_name)
- Pas de --plr_enabled ; CE = --plr_lambda 0.0
- Corrigé: modèle MNIST par défaut = 'mlp'
"""

import argparse
import subprocess
import sys
from pathlib import Path

def build_cmd(
    dataset: str,
    model: str,
    run_dir: Path,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_split: float,
    num_workers: int,
    use_cuda: bool,
    # PLR hyperparams (lambda=0.0 => PLR inactif)
    plr_lambda: float,
    plr_tau: float = 0.3,
    plr_k: int = 2,
    plr_space: str = "input",
    data_dir: Path = Path("data"),
):
    cmd = [
        sys.executable, "-u", "-m", "plr_hilbert.train",
        "--dataset", dataset,
        "--model", model,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--val_split", str(val_split),
        "--data_dir", str(data_dir),
        "--run_dir", str(run_dir),
        "--seed", str(seed),
        "--num_workers", str(num_workers),
        "--plr_lambda", str(plr_lambda),
        "--plr_tau", str(plr_tau),
        "--plr_k", str(plr_k),
        "--plr_space", plr_space,
    ]
    if use_cuda:
        cmd.append("--use_cuda")  # store_true côté trainer
    return cmd

def run_one(cmd, cwd: Path):
    print("\n>>> Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Échec d'entraînement (code {proc.returncode}).")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path("runs") / "update_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-cuda", action="store_true", default=True)

    # MNIST
    p.add_argument("--mnist-epochs", type=int, default=20)
    p.add_argument("--mnist-batch-size", type=int, default=128)
    p.add_argument("--mnist-lr", type=float, default=1e-3)
    p.add_argument("--mnist-weight-decay", type=float, default=5e-4)
    p.add_argument("--mnist-val-split", type=float, default=0.1)
    p.add_argument("--mnist-model", type=str, default="mlp")  # << corrige ici
    p.add_argument("--mnist-plr-lambda", type=float, default=0.6)
    p.add_argument("--mnist-plr-tau", type=float, default=0.30)
    p.add_argument("--mnist-plr-k", type=int, default=2)
    p.add_argument("--mnist-plr-space", type=str, default="input", choices=["input", "feature"])

    # CIFAR-10
    p.add_argument("--cifar-epochs", type=int, default=120)
    p.add_argument("--cifar-batch-size", type=int, default=128)
    p.add_argument("--cifar-lr", type=float, default=1e-3)
    p.add_argument("--cifar-weight-decay", type=float, default=5e-4)
    p.add_argument("--cifar-val-split", type=float, default=0.1)
    p.add_argument("--cifar-model", type=str, default="cnn_small")
    p.add_argument("--cifar-plr-lambda", type=float, default=0.10)
    p.add_argument("--cifar-plr-tau", type=float, default=0.35)
    p.add_argument("--cifar-plr-k", type=int, default=2)
    p.add_argument("--cifar-plr-space", type=str, default="feature", choices=["input", "feature"])

    p.add_argument("--only", type=str, choices=["mnist", "cifar10"], default=None)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "plr_hilbert" / "train.py").exists():
        raise SystemExit("Introuvable: plr_hilbert/train.py (lance depuis la racine du repo).")

    args.runs_root.mkdir(parents=True, exist_ok=True)

    # ---------- MNIST ----------
    if args.only in (None, "mnist"):
        # CE (lambda=0.0)
        run_dir = args.runs_root / f"mnist_{args.mnist_model}_ce_s{args.seed}"
        cmd = build_cmd(
            dataset="mnist",
            model=args.mnist_model,
            run_dir=run_dir,
            seed=args.seed,
            epochs=args.mnist_epochs,
            batch_size=args.mnist_batch_size,
            lr=args.mnist_lr,
            weight_decay=args.mnist_weight_decay,
            val_split=args.mnist_val_split,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,
            plr_lambda=0.0,
            plr_tau=args.mnist_plr_tau,
            plr_k=args.mnist_plr_k,
            plr_space=args.mnist_plr_space,
        )
        run_one(cmd, cwd=repo_root)

        # PLR
        run_dir = args.runs_root / f"mnist_{args.mnist_model}_plr_s{args.seed}"
        cmd = build_cmd(
            dataset="mnist",
            model=args.mnist_model,
            run_dir=run_dir,
            seed=args.seed,
            epochs=args.mnist_epochs,
            batch_size=args.mnist_batch_size,
            lr=args.mnist_lr,
            weight_decay=args.mnist_weight_decay,
            val_split=args.mnist_val_split,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,
            plr_lambda=args.mnist_plr_lambda,
            plr_tau=args.mnist_plr_tau,
            plr_k=args.mnist_plr_k,
            plr_space=args.mnist_plr_space,
        )
        run_one(cmd, cwd=repo_root)

    # ---------- CIFAR-10 ----------
    if args.only in (None, "cifar10"):
        # CE (lambda=0.0)
        run_dir = args.runs_root / f"c10_{args.cifar_model}_ce_s{args.seed}"
        cmd = build_cmd(
            dataset="cifar10",
            model=args.cifar_model,
            run_dir=run_dir,
            seed=args.seed,
            epochs=args.cifar_epochs,
            batch_size=args.cifar_batch_size,
            lr=args.cifar_lr,
            weight_decay=args.cifar_weight_decay,
            val_split=args.cifar_val_split,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,
            plr_lambda=0.0,
            plr_tau=args.cifar_plr_tau,
            plr_k=args.cifar_plr_k,
            plr_space=args.cifar_plr_space,
        )
        run_one(cmd, cwd=repo_root)

        # PLR
        run_dir = args.runs_root / f"c10_{args.cifar_model}_plr_s{args.seed}"
        cmd = build_cmd(
            dataset="cifar10",
            model=args.cifar_model,
            run_dir=run_dir,
            seed=args.seed,
            epochs=args.cifar_epochs,
            batch_size=args.cifar_batch_size,
            lr=args.cifar_lr,
            weight_decay=args.cifar_weight_decay,
            val_split=args.cifar_val_split,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,
            plr_lambda=args.cifar_plr_lambda,
            plr_tau=args.cifar_plr_tau,
            plr_k=args.cifar_plr_k,
            plr_space=args.cifar_plr_space,
        )
        run_one(cmd, cwd=repo_root)

    print("\n✅ Terminé. Les nouveaux runs sont sous:", args.runs_root)

if __name__ == "__main__":
    main()
