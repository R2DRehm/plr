
import os, json, time
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .utils.seed import set_seed
from .utils.logging_utils import ensure_dir, save_config, append_metrics_csv, timestamp
from .data import get_loaders
from .models import MLP, SmallCNN
from .losses.plr import plr_loss
from .losses.calibration import ece_score, brier_score

def build_model(dataset: str, model_name: str, in_shape, num_classes: int):
    model_name = model_name.lower()
    if model_name == "mlp":
        in_dim = 1
        for s in in_shape: in_dim *= s
        return MLP(in_dim=in_dim, num_classes=num_classes, hidden=256, dropout=0.1)
    elif model_name in ["cnn_small", "cnn", "smallcnn"]:
        channels = in_shape[0]
        return SmallCNN(num_classes=num_classes, channels=channels, feat_dim=256, dropout=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())

def train_one_epoch(model, loader, optimizer, device, plr_cfg: Dict, space: str):
    """
    Entraîne un epoch.
    - space == "feature": distances dans l'espace des features penultièmes (recommandé pour CIFAR).
      Par défaut, on calcule ces features sans dropout (plus stable) puis on les L2-normalise.
      (Désactive via plr_cfg["use_nodrop_features"]=False si BatchNorm te pose problème.)
    - space == "input": distances dans l'espace d'entrée (on z-score par échantillon).

    plr_cfg attend (tous optionnels) :
      - "lambda": float (force PLR)
      - "tau": float (seuil)
      - "k": int (nb voisins)
      - "use_nodrop_features": bool (def=True)
      - "normalize_input": bool (def=True, z-score par échantillon)
    """
    model.train()
    ce = nn.CrossEntropyLoss()

    lam = float(plr_cfg.get("lambda", 0.0))
    tau = float(plr_cfg.get("tau", 0.30))
    k   = int(plr_cfg.get("k", 2))
    use_nodrop_features = bool(plr_cfg.get("use_nodrop_features", True))
    normalize_input     = bool(plr_cfg.get("normalize_input", True))

    tot_ce, tot_plr, tot_acc, n_batches = 0.0, 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # 1) Logits (forward "train" normal, pour la CE)
        if space == "feature":
            logits, feat_train = model(x, return_features=True)

            # 2) Espace de distance: features stables (sans dropout) + L2-norm
            if use_nodrop_features:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    _, feat_nodrop = model(x, return_features=True)
                if was_training:
                    model.train()
                Xspace = F.normalize(feat_nodrop, dim=1)
            else:
                # Utilise les features du forward train (inclut dropout) mais détachées
                Xspace = F.normalize(feat_train.detach(), dim=1)

        else:
            # Distances calculées dans l'espace entrée (aplaties)
            logits = model(x)
            flat = x.view(x.size(0), -1)
            if normalize_input:
                mu = flat.mean(dim=1, keepdim=True)
                sigma = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
                flat = (flat - mu) / sigma
            Xspace = flat.detach()

        # 3) CE + PLR (optionnelle)
        loss_ce = ce(logits, y)
        loss = loss_ce
        if lam > 0.0:
            l_plr, plr_info = plr_loss(
                logits=logits,
                X_space=Xspace,
                k=k,
                tau=tau,
                reduction="mean",
            )
            loss = loss + lam * l_plr
            tot_plr += float(l_plr.item())

        # 4) Backprop + step
        loss.backward()
        optimizer.step()

        # 5) Logs
        tot_ce  += float(loss_ce.item())
        tot_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return {
        "train_ce":  tot_ce  / max(1, n_batches),
        "train_plr": tot_plr / max(1, n_batches),
        "train_acc": tot_acc / max(1, n_batches),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    probs_all, labels_all = [], []
    for x, y in loader:
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
    return metrics

def main(
    dataset: str = "mnist",
    model_name: str = "mlp",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    plr_lambda: float = 0.0,
    plr_tau: float = 0.30,
    plr_k: int = 2,
    plr_space: str = "input",  # or "feature"
    val_split: float = 0.1,
    data_dir: str = "./data",
    run_dir: str = None,
    seed: int = 42,
    use_cuda: bool = True,
    num_workers: int = 2,
):
    set_seed(seed)
    loaders = get_loaders(dataset, data_dir, batch_size, val_split, num_workers, seed)
    in_shape = loaders["meta"]["in_shape"]; num_classes = loaders["meta"]["num_classes"]
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = build_model(dataset, model_name, in_shape, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if run_dir is None:
        tag = f"{dataset}_{model_name}_{'plr' if plr_lambda>0 else 'base'}_{timestamp()}"
        run_dir = os.path.join("runs", tag)
    ensure_dir(run_dir)

    cfg = dict(
        dataset=dataset,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        # PLR: paramètres bruts
        plr_lambda=plr_lambda,
        plr_tau=plr_tau,
        plr_k=plr_k,
        plr_space=plr_space,  # "input" ou "feature"
        # PLR: champ dérivé (utile pour l'agrégateur)
        plr_enabled=bool(plr_lambda > 0.0),

        val_split=val_split,
        data_dir=data_dir,
        run_dir=run_dir,
        seed=seed,
        use_cuda=bool(device.type == 'cuda'),
        num_workers=num_workers,
    )

    save_config(run_dir, cfg)

    best_nll = float("inf")
    best_path = os.path.join(run_dir, "best.pt")
    for ep in range(1, epochs+1):
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, device,
            plr_cfg={"lambda": plr_lambda, "tau": plr_tau, "k": plr_k},
            space=plr_space
        )
        val_metrics = evaluate(model, loaders["val"], device)
        row = {"epoch": ep, **train_stats, **{f"val_{k}": v for k,v in val_metrics.items()}}
        append_metrics_csv(run_dir, row)
        print(f"[{ep:03d}/{epochs}] CE:{train_stats['train_ce']:.4f} PLR:{train_stats['train_plr']:.4f} "
              f"Acc:{train_stats['train_acc']:.4f} | Val Acc:{val_metrics['acc']:.4f} "
              f"NLL:{val_metrics['nll']:.4f} ECE:{val_metrics['ece']:.4f} Brier:{val_metrics['brier']:.4f}")
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
    # Save final model
    torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(run_dir, "model.pt"))
    print("Saved:", best_path, "and model.pt")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train with/without PLR")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", type=str, default="mlp")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--plr_lambda", type=float, default=0.0)
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
    main(dataset=args.dataset, model_name=args.model, epochs=args.epochs,
         batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
         plr_lambda=args.plr_lambda, plr_tau=args.plr_tau, plr_k=args.plr_k,
         plr_space=args.plr_space, val_split=args.val_split, data_dir=args.data_dir,
         run_dir=args.run_dir, seed=args.seed, use_cuda=args.use_cuda, num_workers=args.num_workers)
