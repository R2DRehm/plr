
from plr_hilbert.train import main
if __name__ == "__main__":
    # Defaults replicate a quick MNIST MLP run
    main(dataset="mnist", model_name="mlp", epochs=5, batch_size=128,
         lr=1e-3, weight_decay=1e-4, plr_lambda=0.7, plr_tau=0.30, plr_k=2,
         plr_space="input", val_split=0.1, data_dir="./data", run_dir=None,
         seed=42, use_cuda=False, num_workers=2)
