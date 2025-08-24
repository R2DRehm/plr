
import argparse
from plr_hilbert.eval import main as eval_main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--noise_sigma", type=float, default=0.0)
    p.add_argument("--n_bins", type=int, default=15)
    args = p.parse_args()
    eval_main(args.run_dir, ckpt=args.ckpt, noise_sigma=args.noise_sigma, n_bins=args.n_bins)
