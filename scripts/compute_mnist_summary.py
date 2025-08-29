import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try to read overall summary first, else fallback to mnist_aggregate
p = Path('runs/summary.csv')
if p.exists():
    df = pd.read_csv(p)
else:
    p2 = Path('runs/mnist_aggregate.csv')
    if p2.exists():
        df = pd.read_csv(p2)
    else:
        raise SystemExit('No summary CSV found')

# Normalize column names
cols = [c.lower() for c in df.columns]
if 'dataset' not in cols:
    raise SystemExit('CSV has no dataset column')

# lower-case columns map
df.columns = [c.lower() for c in df.columns]

mn = df[df['dataset'] == 'mnist'].copy()
if mn.empty:
    raise SystemExit('No MNIST rows found in CSV')

# Keep only methods we care about: ce, plr with lambda 0.6 and 0.7
# Methods may be stored in 'method' or 'method' like strings
if 'method' not in mn.columns:
    raise SystemExit('no method column')

# Filter numeric ece
mn['ece'] = pd.to_numeric(mn['ece'], errors='coerce')
mn = mn.dropna(subset=['ece']).copy()

# Map plr variants to clean labels
import re

def short_method(m):
    if isinstance(m, float) or isinstance(m, int):
        return str(m)
    m = str(m)
    if m == 'ce' or m.lower() == 'ce':
        return 'ce'
    m_l = re.search(r'l=([0-9\.]+)', m)
    if m_l:
        lam = float(m_l.group(1))
        return f'plr_{lam:.1f}'
    # fallback: keep original
    return m

mn['method_short'] = mn['method'].apply(short_method)

# Keep only ce and plr_0.6 and plr_0.7
want = ['ce','plr_0.6','plr_0.7']
mn_f = mn[mn['method_short'].isin(want)].copy()

if mn_f.empty:
    raise SystemExit('No rows for requested methods: ' + ','.join(want))

grp = mn_f.groupby('method_short')['ece'].agg(['count','mean','std']).reset_index()
grp = grp.sort_values('method_short')

out_csv = Path('runs/mnist_ece_group.csv')
grp.to_csv(out_csv, index=False)
print('Wrote', out_csv)
print(grp.to_string(index=False))

# Plot
plt.figure(figsize=(6,4))
plt.bar(grp['method_short'], grp['mean'], yerr=grp['std'].fillna(0), capsize=5)
plt.ylabel('ECE')
plt.title('MNIST ECE by method (mean ± std)')
plt.tight_layout()
plt.savefig('runs/mnist_ece_group_bar.png', dpi=200)
print('Wrote runs/mnist_ece_group_bar.png')
