#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_runs.py
-----------------
Scanne un dossier 'runs/' contenant des sous-dossiers d'expériences
(p.ex. c10_smallcnn_ce_s0_full_v3, c10_smallcnn_ce_s1_full_v3, ...),
regroupe par configuration en ne variant que la seed (_s\d+),
et calcule moyenne & écart-type des métriques (acc, nll, ece, brier)
sur les seeds. Sauvegarde le tout dans 'res/summary.csv' et 'res/groups.json'.

Usage:
    python scripts/summarize_runs.py \
        --runs-dir runs \
        --out-dir res

Arguments:
    --runs-dir  : répertoire racine des expériences (défaut: runs)
    --out-dir   : répertoire de sortie pour les fichiers (défaut: res)

Conventions supposées:
  - Chaque run a au moins: config.json et metrics_eval.json
  - Le nom de dossier encode la seed sous la forme `_s<chiffre(s)>`
    ex: c10_smallcnn_ce_s3_full_v3  → seed = 3
  - Les runs appartenant au même groupe n'ont que la seed qui change.

Sorties:
  - res/summary.csv  : 1 ligne par groupe (dataset, group_name normalisé)
  - res/groups.json  : détail par groupe (liste des runs + métriques par seed)
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Any, Optional

SEED_PATTERN = re.compile(r"_s(\d+)(?=(_|$))", flags=re.IGNORECASE)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="runs", help="Dossier racine des expériences")
    p.add_argument("--out-dir", type=str, default="res", help="Dossier de sortie")
    return p.parse_args()

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Fichier introuvable: {path}")
    except json.JSONDecodeError as e:
        logging.warning(f"JSON invalide ({path}): {e}")
    return None

def find_seed_in_name(name: str) -> Optional[int]:
    m = SEED_PATTERN.search(name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def normalize_group_name(name: str) -> str:
    """
    Remplace la portion seed par un joker pour définir la clef de regroupement.
    Exemple: 'c10_smallcnn_ce_s3_full_v3' -> 'c10_smallcnn_ce_s*_full_v3'
    """
    return SEED_PATTERN.sub("_s*", name)

def infer_dataset_from_name_or_config(dirname: str, cfg: Optional[Dict[str, Any]]) -> str:
    # 1) Préférence: config.json
    if cfg and "dataset" in cfg and isinstance(cfg["dataset"], str):
        return cfg["dataset"]

    # 2) Sinon: heuristique depuis le nom de dossier
    lower = dirname.lower()
    if lower.startswith("c10") or "cifar10" in lower:
        return "cifar10"
    if lower.startswith("c100") or "cifar100" in lower:
        return "cifar100"
    if lower.startswith("mnist"):
        return "mnist"
    # fallback
    return "unknown"

def discover_run_dirs(runs_root: Path) -> List[Path]:
    """
    Un run valide contient metrics_eval.json (et idéalement config.json).
    On parcourt uniquement le 1er niveau de sous-dossiers de runs_root.
    """
    run_dirs = []
    if not runs_root.exists():
        logging.error(f"Le dossier runs-dir n'existe pas: {runs_root}")
        return run_dirs

    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        metrics = child / "metrics_eval.json"
        if metrics.exists():
            run_dirs.append(child)
    return run_dirs

def safe_stdev(values: List[float]) -> float:
    try:
        return stdev(values) if len(values) >= 2 else 0.0
    except Exception:
        return float("nan")

def float_or_nan(x: Any) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")

def main():
    setup_logging()
    args = parse_args()
    runs_root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        logging.error("Aucun run valide trouvé (metrics_eval.json manquant).")
        return

    # Regroupement: (dataset, group_name_normalisé) -> infos
    groups: Dict[str, Dict[str, Any]] = {}

    for rd in sorted(run_dirs):
        name = rd.name
        cfg = read_json(rd / "config.json") or {}
        metrics = read_json(rd / "metrics_eval.json") or {}

        if not metrics:
            logging.warning(f"Métriques manquantes pour {name}, on saute.")
            continue

        seed = find_seed_in_name(name)
        if seed is None:
            # fallback: seed depuis config.json
            seed = cfg.get("seed", None)

        group_name = normalize_group_name(name)
        dataset = infer_dataset_from_name_or_config(name, cfg)

        key = f"{dataset}::{group_name}"
        g = groups.setdefault(key, {
            "dataset": dataset,
            "group_name": group_name,
            "runs": [],           # liste des runs individuels
            "cfg_prototype": {},  # on capture la première config vue pour contexte
        })

        # Capture de quelques hyperparam utiles (si présents)
        if not g["cfg_prototype"]:
            keep_keys = [
                "model_name", "plr_enabled", "plr_lambda", "plr_tau", "plr_k", "plr_space",
                "epochs", "batch_size", "lr", "weight_decay", "val_split", "data_dir"
            ]
            g["cfg_prototype"] = {k: cfg.get(k, None) for k in keep_keys}

        # Ajoute le run
        g["runs"].append({
            "dir": str(rd),
            "name": name,
            "seed": seed,
            "metrics": {
                "acc": float_or_nan(metrics.get("acc")),
                "nll": float_or_nan(metrics.get("nll")),
                "ece": float_or_nan(metrics.get("ece")),
                "brier": float_or_nan(metrics.get("brier")),
            }
        })

    # Calcul des agrégats
    summary_rows = []
    for key, g in sorted(groups.items(), key=lambda kv: kv[0]):
        metrics_list = [r["metrics"] for r in g["runs"] if r.get("metrics")]
        if not metrics_list:
            continue

        # Vecteurs par métrique
        accs   = [m["acc"]   for m in metrics_list if math.isfinite(m["acc"])]
        nlls   = [m["nll"]   for m in metrics_list if math.isfinite(m["nll"])]
        eces   = [m["ece"]   for m in metrics_list if math.isfinite(m["ece"])]
        briers = [m["brier"] for m in metrics_list if math.isfinite(m["brier"])]

        # Si aucune valeur finie, on met NaN
        def agg(vs: List[float]):
            if vs:
                return mean(vs), safe_stdev(vs), len(vs)
            return float("nan"), float("nan"), 0

        acc_mean,   acc_std,   acc_n   = agg(accs)
        nll_mean,   nll_std,   nll_n   = agg(nlls)
        ece_mean,   ece_std,   ece_n   = agg(eces)
        brier_mean, brier_std, brier_n = agg(briers)

        n_runs = max(acc_n, nll_n, ece_n, brier_n)

        row = {
            "dataset": g["dataset"],
            "group": g["group_name"],
            "n_runs": n_runs,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "nll_mean": nll_mean,
            "nll_std": nll_std,
            "ece_mean": ece_mean,
            "ece_std": ece_std,
            "brier_mean": brier_mean,
            "brier_std": brier_std,
        }

        # Ajoute contexte config principal
        row.update({
            "model_name": g["cfg_prototype"].get("model_name"),
            "plr_enabled": g["cfg_prototype"].get("plr_enabled"),
            "plr_lambda": g["cfg_prototype"].get("plr_lambda"),
            "plr_tau": g["cfg_prototype"].get("plr_tau"),
            "plr_k": g["cfg_prototype"].get("plr_k"),
            "plr_space": g["cfg_prototype"].get("plr_space"),
            "epochs": g["cfg_prototype"].get("epochs"),
            "batch_size": g["cfg_prototype"].get("batch_size"),
            "lr": g["cfg_prototype"].get("lr"),
            "weight_decay": g["cfg_prototype"].get("weight_decay"),
            "val_split": g["cfg_prototype"].get("val_split"),
        })

        summary_rows.append(row)

    # Écrit CSV
    csv_path = out_dir / "summary.csv"
    if summary_rows:
        # Détermine l'ordre des colonnes
        base_cols = [
            "dataset", "group", "n_runs",
            "acc_mean", "acc_std",
            "nll_mean", "nll_std",
            "ece_mean", "ece_std",
            "brier_mean", "brier_std",
        ]
        cfg_cols = [
            "model_name", "plr_enabled", "plr_lambda", "plr_tau", "plr_k", "plr_space",
            "epochs", "batch_size", "lr", "weight_decay", "val_split",
        ]
        cols = base_cols + cfg_cols

        with csv_path.open("w", encoding="utf-8") as f:
            # header
            f.write(",".join(cols) + "\n")

            def fmt(x):
                if isinstance(x, float):
                    # format court mais lisible
                    return f"{x:.6f}"
                return "" if x is None else str(x)

            for row in summary_rows:
                f.write(",".join(fmt(row.get(c)) for c in cols) + "\n")

        logging.info(f"Résumé écrit: {csv_path}")
    else:
        logging.warning("Aucun groupe agrégé; 'summary.csv' non écrit.")

    # Écrit JSON détaillé (groupes)
    groups_json_path = out_dir / "groups.json"
    # on sérialise de manière lisible
    with groups_json_path.open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    logging.info(f"Détails de groupes écrits: {groups_json_path}")

    # Petit rappel console
    logging.info("Terminé.")
    logging.info("Astuce: ouvrez 'res/summary.csv' dans Excel/Sheets, et triez par dataset/group.")

if __name__ == "__main__":
    # Dépend uniquement de la stdlib
    main()
