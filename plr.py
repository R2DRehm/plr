import numpy as np
import pandas as pd
import math, os, json
import matplotlib.pyplot as plt

OUTDIR = "/mnt/data/plr_final"
os.makedirs(OUTDIR, exist_ok=True)

def set_seed(seed:int=0):
    np.random.seed(seed)

def softmax(z):
    zmax = np.max(z, axis=1, keepdims=True)
    ez = np.exp(z - zmax)
    return ez / np.sum(ez, axis=1, keepdims=True)

def ece_score(probs, y, n_bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi if i < n_bins - 1 else conf <= hi + 1e-12)
        if np.any(m):
            ece += (np.mean(m)) * abs(np.mean(acc[m]) - np.mean(conf[m]))
    return float(ece)

def brier_score(probs, y):
    N, K = probs.shape
    Y = np.zeros_like(probs); Y[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((probs - Y)**2, axis=1)))

def reliability_curve(probs, y, n_bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids, accs, confs = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi if i < n_bins - 1 else conf <= hi + 1e-12)
        mids.append((lo + hi) / 2.0)
        if np.any(m):
            accs.append(np.mean(acc[m])); confs.append(np.mean(conf[m]))
        else:
            accs.append(np.nan); confs.append(np.nan)
    return np.array(mids), np.array(accs), np.array(confs)

def generate_dataset(n_train=4000, n_val=800, n_test=2000, d=20, K=6,
                     class_imbalance=True, label_noise=0.10, seed=42):
    set_seed(seed)
    weights = np.linspace(2.0, 1.0, K) if class_imbalance else np.ones(K)
    weights = weights / np.sum(weights)
    centers = np.random.randn(K, d) * 2.0
    cov_scales = 0.7 + 0.6 * np.random.rand(K)
    def sample(n):
        counts = np.random.multinomial(n, weights)
        Xs, ys = [], []
        for k in range(K):
            nk = counts[k]
            if nk == 0: continue
            A = np.diag(0.5 + np.random.rand(d))
            g = np.random.randn(nk, d) @ A
            Xk = centers[k] + cov_scales[k] * g
            yk = np.full(nk, k, dtype=int)
            Xs.append(Xk); ys.append(yk)
        X = np.vstack(Xs); y = np.concatenate(ys)
        idx = np.random.permutation(X.shape[0])
        return X[idx], y[idx]
    Xtr, ytr = sample(n_train)
    Xva, yva = sample(n_val)
    Xte, yte = sample(n_test)
    # label noise train only
    nflip = int(label_noise * ytr.size)
    if nflip > 0:
        idx = np.random.choice(ytr.size, size=nflip, replace=False)
        for i in idx:
            old = ytr[i]
            choices = list(range(K)); choices.remove(old)
            ytr[i] = np.random.choice(choices)
    # standardize
    mu = Xtr.mean(axis=0, keepdims=True); sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
    return (Xtr, ytr), (Xva, yva), (Xte, yte), K, d

def relu(x): return np.maximum(0.0, x)
def relu_grad(h): return (h > 0).astype(h.dtype)

def init_params(d, h, K, seed=0):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(d, h) * (1.0 / np.sqrt(d)); b1 = np.zeros(h)
    W2 = rng.randn(h, K) * (1.0 / np.sqrt(h)); b2 = np.zeros(K)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_mlp(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Hpre = X @ W1 + b1; H = relu(Hpre); Z = H @ W2 + b2; P = softmax(Z)
    cache = {"X": X, "Hpre": Hpre, "H": H, "Z": Z, "P": P}
    return P, Z, cache

def nll_grad_ce_from_cache(y, cache):
    P, Z = cache["P"], cache["Z"]
    B = P.shape[0]
    Y = np.zeros_like(P); Y[np.arange(B), y] = 1.0
    nll = -np.mean(np.log(P[np.arange(B), y] + 1e-12))
    dZ = (P - Y) / B
    return nll, dZ

def backprop_from_dZ(dZ, cache, params, wd=1e-4):
    X, H = cache["X"], cache["H"]
    W1, W2 = params["W1"], params["W2"]
    dW2 = H.T @ dZ + wd * W2; db2 = np.sum(dZ, axis=0)
    dH = dZ @ W2.T; dHpre = dH * relu_grad(H)
    dW1 = X.T @ dHpre + wd * W1; db1 = np.sum(dHpre, axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def adam_update(param, grad, state, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m, v, t = state; t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    mhat = m / (1 - beta1 ** t); vhat = v / (1 - beta1 ** t)
    param -= lr * mhat / (np.sqrt(vhat) + eps)
    state[:] = [m, v, t]
    return param

def knn_pairs_indices(Xb, k=2):
    B = Xb.shape[0]
    diff = Xb[:, None, :] - Xb[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    idx = np.argpartition(dist2, kth=k-1, axis=1)[:, :k]
    i_idx = np.repeat(np.arange(B), k)
    j_idx = idx.reshape(-1)
    return i_idx, j_idx

def plr_penalty_and_grad(Xb, Zb, i_idx, j_idx, tau):
    Pairs = i_idx.shape[0]
    dX = Xb[i_idx] - Xb[j_idx]
    dx = np.sqrt(np.sum(dX * dX, axis=1) + 1e-12)
    scale = np.median(dx); dx_scaled = dx / (scale + 1e-12)
    dz = Zb[i_idx] - Zb[j_idx]
    rng = dz.max(axis=1) - dz.min(axis=1)
    ratios = rng / (dx_scaled + 1e-12)
    over = ratios - tau
    active = over > 0.0
    penalty = np.mean(np.maximum(0.0, over))
    G = np.zeros_like(dz)
    if np.any(active):
        rows = np.nonzero(active)[0]
        imax = np.argmax(dz[active], axis=1)
        imin = np.argmin(dz[active], axis=1)
        invdx = 1.0 / (dx_scaled[active] + 1e-12)
        for rr, im, ii, inv in zip(rows, imax, imin, invdx):
            G[rr, im] += inv; G[rr, ii] -= inv
    G /= float(Pairs)
    return penalty, G, active

def train_one(Xtr, ytr, Xva, yva, d, K, seed, lam_max=0.7, tau=0.30, k=2,
              warmup=10, ramp=25, epochs=50, batch_size=128, lr=3e-3, wd=1e-4):
    np.random.seed(seed)
    params = init_params(d, 64, K, seed=seed)
    st = {"W1":[np.zeros_like(params["W1"]), np.zeros_like(params["W1"]), 0],
          "b1":[np.zeros_like(params["b1"]), np.zeros_like(params["b1"]), 0],
          "W2":[np.zeros_like(params["W2"]), np.zeros_like(params["W2"]), 0],
          "b2":[np.zeros_like(params["b2"]), np.zeros_like(params["b2"]), 0]}
    def lam_at_epoch(ep):
        if ep <= warmup: return 0.0
        t = min(1.0, (ep - warmup) / float(max(1, ramp)))
        return lam_max * 0.5 * (1 - math.cos(math.pi * t))
    N = Xtr.shape[0]; nb = int(math.ceil(N / batch_size))
    hist = {"lam":[], "plr":[], "nll_val":[]}
    for ep in range(1, epochs+1):
        perm = np.random.permutation(N); Xs, ys = Xtr[perm], ytr[perm]
        lam = lam_at_epoch(ep)
        plr_vals = []
        for bi in range(nb):
            s, e = bi*batch_size, min(N, (bi+1)*batch_size)
            Xb, yb = Xs[s:e], ys[s:e]
            P, Z, cache = forward_mlp(Xb, params)
            nll, dZ = nll_grad_ce_from_cache(yb, cache)
            if lam > 0.0 and Xb.shape[0] > 2:
                i_idx, j_idx = knn_pairs_indices(Xb, k=k)
                pen, G, active = plr_penalty_and_grad(Xb, Z, i_idx, j_idx, tau)
                if np.any(active):
                    rows = np.nonzero(active)[0]
                    i_act, j_act = i_idx[rows], j_idx[rows]
                    G_act = G[rows]
                    dZ[i_act] += lam * G_act
                    dZ[j_act] -= lam * G_act
                plr_vals.append(float(pen))
            grads = backprop_from_dZ(dZ, cache, params, wd=wd)
            params["W2"] = adam_update(params["W2"], grads["dW2"], st["W2"], lr=lr)
            params["b2"] = adam_update(params["b2"], grads["db2"], st["b2"], lr=lr)
            params["W1"] = adam_update(params["W1"], grads["dW1"], st["W1"], lr=lr)
            params["b1"] = adam_update(params["b1"], grads["db1"], st["b1"], lr=lr)
        # val
        Pv, Zv, _ = forward_mlp(Xva, params)
        nll_val = -np.mean(np.log(Pv[np.arange(Pv.shape[0]), yva] + 1e-12))
        hist["lam"].append(float(lam)); hist["plr"].append(float(np.mean(plr_vals) if plr_vals else 0.0)); hist["nll_val"].append(float(nll_val))
    return params, hist

def evaluate_all(X, y, params, noise_sigma=None):
    Xe = X.copy()
    if noise_sigma is not None and noise_sigma > 0:
        Xe = Xe + noise_sigma * np.random.randn(*Xe.shape)
    P, Z, _ = forward_mlp(Xe, params)
    nll = -np.mean(np.log(P[np.arange(P.shape[0]), y] + 1e-12))
    acc = float(np.mean(np.argmax(P, axis=1) == y))
    ece = ece_score(P, y); brier = brier_score(P, y)
    return {"acc": acc, "nll": float(nll), "ece": ece, "brier": brier}, P, Z

def mean_projective_ratio(X, Z, k=2, samples=1200, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(samples, X.shape[0]), replace=False)
    Xb, Zb = X[idx], Z[idx]
    i_idx, j_idx = knn_pairs_indices(Xb, k=k)
    dX = Xb[i_idx] - Xb[j_idx]
    dx = np.sqrt(np.sum(dX * dX, axis=1) + 1e-12); scale = np.median(dx)
    dz = Zb[i_idx] - Zb[j_idx]
    rng_logits = dz.max(axis=1) - dz.min(axis=1)
    ratios = rng_logits / (dx / (scale + 1e-12))
    return float(np.mean(ratios)), float(np.median(ratios))

# Prepare data
(Xtr, ytr), (Xva, yva), (Xte, yte), K, d = generate_dataset(seed=999)

rows = []

# Seed 0 only first (baseline + PLR) to ensure completion
for seed in [0]:
    # Baseline
    params0, hist0 = train_one(Xtr, ytr, Xva, yva, d, K, seed=seed, lam_max=0.0)
    m0_clean, P0, Z0 = evaluate_all(Xte, yte, params0, noise_sigma=None)
    m0_noisy, _, _ = evaluate_all(Xte, yte, params0, noise_sigma=0.25)
    mr0_mean, mr0_med = mean_projective_ratio(Xte, Z0, seed=seed)
    rows.append({"seed": seed, "variant":"baseline", "sigma":0.0, **m0_clean, "proj_ratio_mean": mr0_mean})
    rows.append({"seed": seed, "variant":"baseline", "sigma":0.25, **m0_noisy, "proj_ratio_mean": mr0_mean})

    # PLR
    params1, hist1 = train_one(Xtr, ytr, Xva, yva, d, K, seed=seed, lam_max=0.7, tau=0.30, k=2,
                               warmup=10, ramp=25, epochs=50)
    m1_clean, P1, Z1 = evaluate_all(Xte, yte, params1, noise_sigma=None)
    m1_noisy, _, _ = evaluate_all(Xte, yte, params1, noise_sigma=0.25)
    mr1_mean, mr1_med = mean_projective_ratio(Xte, Z1, seed=seed)
    rows.append({"seed": seed, "variant":"plr", "sigma":0.0, **m1_clean, "proj_ratio_mean": mr1_mean})
    rows.append({"seed": seed, "variant":"plr", "sigma":0.25, **m1_noisy, "proj_ratio_mean": mr1_mean})

    # Save reliability plot for this seed
    mids0, accs0, _ = reliability_curve(P0, yte, n_bins=15)
    mids1, accs1, _ = reliability_curve(P1, yte, n_bins=15)
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    m0 = ~np.isnan(accs0); m1 = ~np.isnan(accs1)
    plt.plot(mids0[m0], accs0[m0], marker="o", label="Baseline")
    plt.plot(mids1[m1], accs1[m1], marker="o", label="PLR")
    plt.xlabel("Confidence (mean per bin)"); plt.ylabel("Empirical accuracy (per bin)")
    plt.title("Reliability (Test Clean) — Seed 0")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "reliability_seed0.png"), dpi=150); plt.close()

    # Save hist for seed 0
    with open(os.path.join(OUTDIR, "hist_seed0.json"), "w") as f:
        json.dump({"baseline": hist0, "plr": hist1}, f, indent=2)

# Aggregate and save
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "partial_results_seed0.csv"), index=False)

print(df)