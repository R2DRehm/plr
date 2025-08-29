"""Script minimal pour lancer des expériences MNIST propres (CE vs PLR) sur plusieurs seeds.
Usage: python scripts/run_mnist_experiments.py

Il lance des runs en appelant run_train.main via subprocess (CLI) pour garder l'environnement identique.
Les runs sont taggés avec suffix `_clean` pour être capturés par l'agrégateur.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = 'plr-gpu'

# paramètres
seeds = [0,1,2,3,4]
epochs = 5
batch_size = 128

runs = []
for s in seeds:
    # CE baseline
    tag_ce = f"mnist_mlp_ce_s{s}_clean"
    cmd_ce = ['conda','run','-n',ENV,'python','run_train.py',
              "--dataset", "mnist", "--model", "mlp",
              "--epochs", str(epochs), "--batch_size", str(batch_size),
              "--plr_lambda", "0.0", "--seed", str(s), "--run_dir", f"runs/{tag_ce}", "--use_cuda"]
    runs.append(cmd_ce)

    # PLR
    tag_plr = f"mnist_mlp_plr_s{s}_clean"
    cmd_plr = ['conda','run','-n',ENV,'python','run_train.py',
               "--dataset", "mnist", "--model", "mlp",
               "--epochs", str(epochs), "--batch_size", str(batch_size),
               "--plr_lambda", "0.6", "--plr_tau", "0.30", "--plr_k", "2",
               "--plr_space", "input", "--seed", str(s), "--run_dir", f"runs/{tag_plr}", "--use_cuda"]
    runs.append(cmd_plr)

    # small pause can be added if needed

# Execution séquentielle
for cmd in runs:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

print("All MNIST experiments finished.")
