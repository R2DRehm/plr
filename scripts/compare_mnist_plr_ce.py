import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import csv

# List of PLR run dirs you provided
plr_dirs = [
"runs/mnist_mlp_plr_20250829-120429",
"runs/mnist_mlp_plr_20250829-120523",
"runs/mnist_mlp_plr_20250829-120615",
"runs/mnist_mlp_plr_20250829-120707",
"runs/mnist_mlp_plr_20250829-120759",
"runs/mnist_mlp_plr_20250829-120852",
"runs/mnist_mlp_plr_20250829-120944",
"runs/mnist_mlp_plr_20250829-121036",
"runs/mnist_mlp_plr_20250829-121128",
"runs/mnist_mlp_plr_20250829-121219",
"runs/mnist_mlp_plr_20250829-122618",
"runs/mnist_mlp_plr_20250829-122711",
"runs/mnist_mlp_plr_20250829-122805",
"runs/mnist_mlp_plr_20250829-122859",
"runs/mnist_mlp_plr_20250829-122952",
]

# CE runs we created earlier
ce_dirs = [f"runs/mnist_mlp_ce_s{i}_clean" for i in range(5)]

all_dirs = [(d, 'plr') for d in plr_dirs] + [(d, 'ce') for d in ce_dirs]
rows = []

def read_cfg(run):
    p = Path(run) / 'config.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))

def read_metrics(run):
    r = Path(run)
    # prefer eval JSON
    for name in ['metrics_eval.json','eval.json','metrics_eval.csv','eval.csv','metrics.csv','metrics_epoch.csv']:
        p = r / name
        if p.exists():
            try:
                if p.suffix == '.json':
                    return json.loads(p.read_text(encoding='utf-8'))
                elif p.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(p)
                    row = df.iloc[-1].to_dict()
                    return row
            except Exception:
                pass
    # fallback: return empty
    return {}

for d, typ in all_dirs:
    p = Path(d)
    if not p.exists():
        print('Missing', d)
        continue
    cfg = read_cfg(d)
    m = read_metrics(d)
    # determine method label
    lam = cfg.get('plr_lambda', None)
    method = 'ce' if (lam is None or float(lam) <= 0.0) else f'plr_{float(lam):.1f}'
    # seed
    seed = cfg.get('seed', None)
    ece = None
    if isinstance(m, dict):
        # try keys
        for k in ['ece','val_ece']:
            if k in m:
                try:
                    ece = float(m[k])
                    break
                except Exception:
                    pass
    rows.append({'run': p.name, 'path': str(p.resolve()), 'method': method, 'seed': seed, 'ece': ece})

if not rows:
    print('No runs parsed')
    raise SystemExit(1)

df = pd.DataFrame(rows)
df.to_csv('runs/mnist_plr_vs_ce.csv', index=False)
print('Wrote runs/mnist_plr_vs_ce.csv')
print(df)

# aggregate by method
agg = df.groupby('method')['ece'].agg(['count','mean','std']).reset_index()
agg.to_csv('runs/mnist_plr_vs_ce_agg.csv', index=False)
print('Wrote runs/mnist_plr_vs_ce_agg.csv')
print(agg)

# simple barplot
plt.figure(figsize=(6,4))
plt.bar(agg['method'], agg['mean'], yerr=agg['std'].fillna(0), capsize=5)
plt.ylabel('ECE')
plt.title('MNIST ECE: PLR vs CE')
plt.tight_layout()
plt.savefig('runs/mnist_plr_vs_ce.png', dpi=200)
print('Wrote runs/mnist_plr_vs_ce.png')
