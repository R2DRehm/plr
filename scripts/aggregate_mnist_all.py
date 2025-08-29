"""Aggregate all MNIST runs under runs/ and runs/archive/, create CSV and ECE barplot.
Usage: python scripts/aggregate_mnist_all.py
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
RUNS = list(ROOT.glob('runs/**/config.json'))
rows = []
for cfgp in RUNS:
    run_dir = cfgp.parent
    try:
        cfg = json.loads(cfgp.read_text(encoding='utf-8'))
    except Exception:
        continue
    ds = cfg.get('dataset','').lower()
    if ds != 'mnist':
        continue
    # find metrics
    metrics = {}
    for name in ['metrics_eval.json','eval.json','metrics_eval.csv','metrics_epoch.csv','metrics.csv']:
        f = run_dir / name
        if not f.exists():
            continue
        try:
            if f.suffix == '.json':
                metrics = json.loads(f.read_text(encoding='utf-8'))
            else:
                df = pd.read_csv(f)
                # take last row
                metrics = df.iloc[-1].to_dict()
        except Exception:
            continue
        if metrics:
            break
    # normalize metric keys
    if not metrics:
        continue
    # prefer keys acc,nll,ece,brier
    m = {}
    for k in ['acc','nll','ece','brier']:
        if k in metrics:
            try:
                m[k] = float(metrics[k])
            except Exception:
                m[k] = None
        else:
            # try val_ prefix
            if ('val_'+k) in metrics:
                try:
                    m[k] = float(metrics['val_'+k])
                except Exception:
                    m[k] = None
            else:
                m[k] = None
    # method tag
    plr_enabled = bool(cfg.get('plr_enabled', cfg.get('plr_lambda', 0.0) > 0.0))
    if plr_enabled:
        method = f"plr(l={cfg.get('plr_lambda')},t={cfg.get('plr_tau')},k={cfg.get('plr_k')})"
    else:
        method = 'ce'
    seed = cfg.get('seed', -1)
    rows.append({
        'run': run_dir.name,
        'path': str(run_dir),
        'dataset': ds,
        'method': method,
        'seed': seed,
        **m
    })

if not rows:
    print('No MNIST runs found.')
    raise SystemExit(0)

df = pd.DataFrame(rows)
df.to_csv('runs/mnist_aggregate.csv', index=False)
print('Wrote runs/mnist_aggregate.csv with', len(df), 'rows')

# aggregate
agg = df.groupby('method')[['acc','nll','ece','brier']].agg(['mean','std','count'])
print(agg)

# ECE barplot mean±std
try:
    plot_df = df.groupby('method')['ece'].agg(['mean','std']).reset_index()
    plt.figure(figsize=(6,4))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df['mean'], yerr=plot_df['std'], capsize=4)
    plt.xticks(x, plot_df['method'], rotation=30)
    plt.ylabel('ECE')
    plt.title('MNIST — ECE mean ± std')
    plt.tight_layout()
    plt.savefig('runs/mnist_ece_compare.png', dpi=200)
    plt.close()
    print('Wrote runs/mnist_ece_compare.png')
except Exception as e:
    print('Plot error:', e)

# print seed-wise table
print('\nSeed-wise:')
print(df[['run','method','seed','acc','nll','ece','brier']])
