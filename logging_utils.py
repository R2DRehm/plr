
import json, os, csv, datetime
from typing import Dict, List, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def save_config(run_dir: str, cfg: Dict[str, Any]):
    ensure_dir(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

def load_config(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        return json.load(f)

def append_metrics_csv(run_dir: str, row: Dict[str, Any], filename: str = "metrics_epoch.csv"):
    path = os.path.join(run_dir, filename)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_final_metrics(run_dir: str, rows: List[Dict[str, Any]], filename: str = "metrics.csv"):
    ensure_dir(run_dir)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(run_dir, filename), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(rows[-1], f, indent=2)
