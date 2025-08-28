# scripts/aggregate_and_plots.py
import re, subprocess, sys, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Quels runs lire ? (ajoute facilement d'autres motifs si besoin)
# --- Quels runs lire ? (ajoute facilement d'autres motifs si besoin)
PATTERNS = [
    # CIFAR-10
    "c10_smallcnn_ce_s*_full*",
    "c10_smallcnn_plr_t0p35_s*_full*",
    "c10_smallcnn_plr_l0p4_s*_full*",
    "c10_smallcnn_plr_s*_full*",

    # CIFAR-100
    "c100_smallcnn_ce_s*_full*",
    "c100_smallcnn_plr_t0p35_s*_full*",
]



RUNS = []
for pat in PATTERNS:
    RUNS += [p for p in Path("runs").glob(pat) if p.is_dir()]
RUNS = sorted(RUNS)


def read_cfg(run: Path) -> dict:
    cfg = {}
    cfg_path = run / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    return cfg

def load_cfg(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def tag_method_from_cfg(cfg: dict) -> str:
    # Fallback si vieux runs sans plr_enabled: on infère depuis plr_lambda
    plr_enabled = bool(cfg.get("plr_enabled", cfg.get("plr_lambda", 0.0) > 0.0))
    if not plr_enabled:
        return "ce"

    lam = cfg.get("plr_lambda", None)
    tau = cfg.get("plr_tau", None)
    k   = cfg.get("plr_k", None)
    space = cfg.get("plr_space", "input")
    space_tag = "f" if space == "feature" else "i"
    # Ex: plr(l=0.6,t=0.35,k=2,f)
    return f"plr(l={lam},t={tau},k={k},{space_tag})"

def parse_or_eval(run: Path):
    # 1) Essaie quelques fichiers d'évaluation courants
    for name in ["metrics_eval.json", "eval.json", "eval.csv", "metrics_eval.csv", "eval_metrics.csv", "metrics.csv"]:
        f = run / name
        if not f.exists():
            continue
        try:
            if f.suffix == ".csv":
                df = pd.read_csv(f)
                row = df.iloc[-1].to_dict()
            else:
                with open(f, "r", encoding="utf-8") as jf:
                    row = json.load(jf)
            if {"acc","nll","ece","brier"}.issubset(row.keys()):
                return {k: float(row[k]) for k in ["acc","nll","ece","brier"]}
        except Exception:
            pass
    # 2) Sinon, appelle l'éval (stdout JSON en fin)
    out = subprocess.check_output([sys.executable, "-m", "plr_hilbert.eval", "--run", str(run)], text=True)
    def grab(k):
        m = re.search(rf'"{k}"\s*:\s*([0-9.]+)', out)
        return float(m.group(1)) if m else None
    return {"acc":grab("acc"), "nll":grab("nll"), "ece":grab("ece"), "brier":grab("brier")}


def method_from_config(run_dir: Path, cfg: dict) -> str:
    lam = float(cfg.get("plr_lambda", 0.0))
    if lam <= 0.0:
        return "ce"
    tau = float(cfg.get("plr_tau", 0.0))
    k   = int(cfg.get("plr_k", 0))
    space = cfg.get("plr_space", "input")
    s_short = "f" if space.lower().startswith("feat") else "i"
    return f"plr(l={lam:.1f},t={tau:.2f},k={k},{s_short})"

def dataset_from_config(run_dir: Path, cfg: dict) -> str:
    ds = cfg.get("dataset", "").lower()
    if ds:
        return ds
    # fallback léger par nom de dossier
    n = run_dir.name.lower()
    if n.startswith("c100_") or "c100_" in n: return "cifar100"
    if n.startswith("c10_")  or "c10_"  in n: return "cifar10"
    if "fashion" in n:                         return "fashion-mnist"
    if "mnist" in n:                           return "mnist"
    return "unknown"

def seed_from_name(runname: str) -> int:
    m = re.search(r"_s(\d+)_", runname)
    return int(m.group(1)) if m else -1

if not RUNS:
    print("Aucun run trouvé (vérifie PATTERNS).")
    sys.exit(0)

rows = []
for run in RUNS:
    # 1) Lire le config du run
    cfg = load_cfg(run)
    dataset = cfg.get("dataset", "unknown")

    # 2) Récupérer les métriques (CSV si présent, sinon eval à la volée)
    metrics = parse_or_eval(run)
    if any(v is None for v in metrics.values()):
        print(f"[warn] métriques manquantes pour {run.name}, on ignore.")
        continue

    # 3) Méthode = à partir du config
    method = tag_method_from_cfg(cfg)

    # 4) Seed par le nom (pratique), fallback -1 si absent
    m_seed = re.search(r"_s(\d+)_", run.name)
    seed = int(m_seed.group(1)) if m_seed else -1

    rows.append({
        "run": run.name,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        **metrics
    })


if not rows:
    print("Aucun run exploitable.")
    sys.exit(0)

df = pd.DataFrame(rows).sort_values(["dataset","method","seed"]).reset_index(drop=True)
metric_cols = ["acc","nll","ece","brier"]
for c in metric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=metric_cols).copy()

print("\nSeed-wise:\n", df)

# --- Agrégation: mean, std, IC95% par (dataset, method)
def tcrit(n):
    table = {3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,
             11:2.228,12:2.201,13:2.179,14:2.160,15:2.145,16:2.131,17:2.120,18:2.110,19:2.101,20:2.093}
    return table.get(n, 1.96)

df = df[(df["acc"] > 0.05) & (df["nll"] < 4.0)].copy()

g = df.groupby(["dataset","method"])[metric_cols].agg(["mean","std","count"])

flat = []
for (dataset, method), stats in g.iterrows():
    row = {"dataset": dataset, "method": method}
    for col in metric_cols:
        n  = int(stats[(col, "count")])
        mu = float(stats[(col, "mean")])
        sd = float(stats[(col, "std")])
        ci = tcrit(n) * (sd / max(np.sqrt(n), 1))
        row[f"{col}_mean"] = mu
        row[f"{col}_std"]  = sd
        row[f"{col}_n"]    = n
        row[f"{col}_ci95"] = ci
    flat.append(row)

summ = pd.DataFrame(flat).sort_values(["dataset","method"]).reset_index(drop=True)
Path("runs").mkdir(exist_ok=True)
summ.to_csv("runs/summary_ci.csv", index=False)

print("\nSummary (mean±std, CI95 in CSV):\n",
      summ[["dataset","method","acc_mean","acc_std","nll_mean","nll_std","ece_mean","ece_std","brier_mean","brier_std"]])

# --- Wilcoxon CE vs PLR(t=0.35) par dataset sur seeds communs
stats_lines = []
try:
    from scipy.stats import wilcoxon
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        ce = sub[sub.method == "ce"].set_index("seed")[metric_cols]
        # Choix de la variante PLR à comparer : priorise t=0.35, sinon prend n'importe quel "plr("
        plr_methods = [m for m in sub["method"].unique() if m.startswith("plr(")]
        target = None
        for m in plr_methods:
            if "t=0.35" in m:
                target = m
                break
        if target is None and plr_methods:
            target = sorted(plr_methods)[0]
        if target is None or ce.empty:
            stats_lines.append(f"[{ds}] pas de comparaison CE vs PLR possible (données manquantes).")
            continue
        pl = sub[sub.method == target].set_index("seed")[metric_cols]
        common = ce.index.intersection(pl.index)
        if len(common) < 5:
            stats_lines.append(f"[{ds}] seeds communs insuffisants pour Wilcoxon (n={len(common)}).")
            continue
        for col in metric_cols:
            s = wilcoxon(ce.loc[common, col], pl.loc[common, col], alternative="two-sided", zero_method="wilcox")
            stats_lines.append(f"[{ds}] Wilcoxon CE vs {target} on {col}: n={len(common)}, W={s.statistic}, p={s.pvalue:.3e}")
except Exception as e:
    stats_lines.append(f"[info] scipy absent ou indisponible pour Wilcoxon ({e}). Installe: pip install scipy")

with open("runs/stats_tests.txt","w",encoding="utf-8") as f:
    f.write("\n".join(stats_lines))
print("\n".join(stats_lines))

# --- Barplots (mean ± std) par dataset
def barplot(subsumm: pd.DataFrame, metric: str, fname: str, ylabel: str, title_prefix: str):
    plt.figure()
    methods = subsumm["method"].tolist()
    means = subsumm[f"{metric}_mean"].to_numpy()
    stds  = subsumm[f"{metric}_std"].to_numpy()
    x = np.arange(len(methods))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=15)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} — {ylabel} (mean ± std)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

for ds in sorted(summ["dataset"].unique()):
    subsumm = summ[summ["dataset"] == ds]
    if subsumm.empty: 
        continue
    prefix = f"runs/{ds}"
    barplot(subsumm, "ece",  f"{prefix}_ece_bar.png",  "ECE (↓)",  ds.upper())
    barplot(subsumm, "nll",  f"{prefix}_nll_bar.png",  "NLL (↓)",  ds.upper())

print("\nFichiers écrits : runs\\summary_ci.csv, runs\\stats_tests.txt, runs\\<dataset>_ece_bar.png, runs\\<dataset>_nll_bar.png")
