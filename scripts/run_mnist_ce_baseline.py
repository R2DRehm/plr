"""Launch MNIST CE baseline runs (plr_lambda=0) for multiple seeds using conda env 'plr-gpu'.
Creates run dirs named runs/mnist_mlp_ce_s{seed}_clean
"""
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ENV = 'plr-gpu'
seeds = [0,1,2,3,4]
epochs = 5
batch_size = 128

ROOT = Path(__file__).resolve().parents[1]

for s in seeds:
    run_dir = f"runs/mnist_mlp_ce_s{s}_clean"
    cmd = [
        'conda', 'run', '-n', ENV, 'python', 'run_train.py',
        '--dataset', 'mnist', '--model', 'mlp',
        '--epochs', str(epochs), '--batch_size', str(batch_size),
        '--plr_lambda', '0.0', '--seed', str(s), '--run_dir', run_dir,
        '--use_cuda'
    ]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

print('All CE baseline runs finished.')
