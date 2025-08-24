
import os, json
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.logging_utils import load_config, ensure_dir
from .data import get_loaders
from .models import MLP, SmallCNN
from .losses.calibration import ece_score, brier_score, reliability_curve, plot_reliability

def build_model(cfg: Dict):
    dataset = cfg["dataset"]; model_name = cfg["model_name"]
    loaders = get_loaders(dataset, cfg["data_dir"], cfg["batch_size"], cfg["val_split"], cfg["num_workers"], cfg["seed"])
    in_shape = loaders["meta"]["in_shape"]; num_classes = loaders["meta"]["num_classes"]
    if model_name == "mlp":
        in_dim = 1
        for s in in_shape: in_dim *= s
        model = MLP(in_dim, num_classes, hidden=256, dropout=0.0)
    else:
        model = SmallCNN(num_classes=num_classes, channels=in_shape[0], feat_dim=256, dropout=0.0)
    return model, loaders

@torch.no_grad()
def evaluate(model, loader, device, noise_sigma: float = 0.0):
    ce = nn.CrossEntropyLoss()
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    probs_all, labels_all = [], []
    for x, y in loader:
        if noise_sigma and noise_sigma > 0:
            x = x + noise_sigma * torch.randn_like(x)
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

    metrics = {
        "acc": tot_acc / max(1, n),
        "nll": tot_ce / max(1, len(loader)),
        "ece": ece_score(probs_all, labels_all),
        "brier": brier_score(probs_all, labels_all),
    }
    return metrics, probs_all, labels_all

def main(run_dir: str, ckpt: str = "best.pt", noise_sigma: float = 0.0, n_bins: int = 15):
    cfg = load_config(run_dir)
    device = torch.device("cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available() else "cpu")
    model, loaders = build_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run_dir, ckpt), map_location=device)["model"])
    model.to(device).eval()

    metrics, probs, labels = evaluate(model, loaders["test"], device, noise_sigma=noise_sigma)
    with open(os.path.join(run_dir, "metrics_eval.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Reliability diagram
    mids, accs, confs = reliability_curve(probs, labels, n_bins=n_bins)
    plot_reliability(mids, accs, confs, title=f"Reliability (sigma={noise_sigma})",
                     out_path=os.path.join(run_dir, "reliability.png"))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Evaluate a trained model and plot reliability.")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--noise_sigma", type=float, default=0.0)
    p.add_argument("--n_bins", type=int, default=15)
    args = p.parse_args()
    main(args.run_dir, ckpt=args.ckpt, noise_sigma=args.noise_sigma, n_bins=args.n_bins)
