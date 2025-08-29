
from plr_hilbert.train import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convenience entrypoint to call plr_hilbert.train.main with CLI args")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", dest="model_name", type=str, default="mlp")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--plr_lambda", type=float, default=0.7)
    p.add_argument("--plr_tau", type=float, default=0.30)
    p.add_argument("--plr_k", type=int, default=2)
    p.add_argument("--plr_space", type=str, default="input", choices=["input","feature"]) 
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)

    args = p.parse_args()

    main(dataset=args.dataset, model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, weight_decay=args.weight_decay, plr_lambda=args.plr_lambda, plr_tau=args.plr_tau, plr_k=args.plr_k,
         plr_space=args.plr_space, val_split=args.val_split, data_dir=args.data_dir, run_dir=args.run_dir,
         seed=args.seed, use_cuda=args.use_cuda, num_workers=args.num_workers)
