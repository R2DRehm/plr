# Hilbert-ML — Projective-Lipschitz Regularization (PLR)

Regularize the **sensitivity of class probabilities in the geometry of the simplex** (Hilbert’s projective metric) to improve **calibration**, **NLL**, **Brier score**, and **local robustness** with minimal overhead. PLR is plug-and-play: add one penalty term to your training objective; no architectural changes, no post-hoc calibration.

---

## Contents

- [Motivation](#motivation)
- [Theory (precise)](#theory-precise)
  - [Setup](#setup)
  - [Hilbert projective distance on the simplex](#hilbert-projective-distance-on-the-simplex)
  - [Softmax–Hilbert identity](#softmaxhilbert-identity)
  - [Projective Lipschitz constant](#projective-lipschitz-constant)
  - [Local differential view](#local-differential-view)
- [Method: PLR penalties](#method-plr-penalties)
  - [Variant A — Pairs *k*-NN (finite differences)](#variant-a--pairs-k-nn-finite-differences)
  - [Variant B — Jacobian–vector proxy (Jv)](#variant-b--jacobianvector-proxy-jv)
  - [Smoothed range (log-sum-exp)](#smoothed-range-log-sum-exp)
  - [Full training objective](#full-training-objective)
  - [Properties](#properties)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [Run commands](#run-commands)
  - [Example configuration](#example-configuration)
- [Evaluation & calibration metrics](#evaluation--calibration-metrics)
  - [Reliability diagram](#reliability-diagram)
- [Hyperparameters & complexity](#hyperparameters--complexity)
- [Roadmap (paper-ready)](#roadmap-paper-ready)
- [Tests](#tests)
- [Citation](#citation)
- [License & acknowledgments](#license--acknowledgments)

---

## Motivation

Modern classifiers are often **over-confident**. Common fixes (temperature scaling, label smoothing) either operate **post-hoc** or in an **Euclidean logit space**. However, probabilities live on the **probability simplex**. The **Hilbert projective metric** is the right geometry to control **odds variations**. PLR directly regularizes the **projective Lipschitz constant** of the probability map \\(x \\mapsto p_\\theta(x)\\), yielding better calibration with small, interpretable overhead.

---

## Theory (precise)

### Setup

- Input metric space \\((\\mathcal{X}, \\|\\cdot\\|)\\).
- Logits \\(z_\\theta : \\mathcal{X} \\to \\mathbb{R}^K\\).
- Probabilities \\(p_\\theta(x) = \\mathrm{softmax}(z_\\theta(x)) \\in \\Delta_{K-1}^{\\circ}\\) (interior of the simplex).

### Hilbert projective distance on the simplex

For \\(p,q \\in \\Delta_{K-1}^{\\circ}\\),
\\[
d_H(p,q)
= \\log\\!\\Big(\\max_i \\frac{p_i}{q_i}\\Big)
 - \\log\\!\\Big(\\min_i \\frac{p_i}{q_i}\\Big)
= \\big[\\max_i(\\log p_i - \\log q_i)\\big]
  - \\big[\\min_i(\\log p_i - \\log q_i)\\big].
\\]

**Interpretation.** \\(d_H(p,q)\\) is the **maximum change over all log-odds**:
\\[
d_H(p,q) \\;=\\; \\max_{i,j} \\left| \\log\\frac{p_i}{p_j} - \\log\\frac{q_i}{q_j} \\right|.
\\]

### Softmax–Hilbert identity

Let \\(p=\\mathrm{softmax}(z)\\), \\(q=\\mathrm{softmax}(z')\\) with \\(z,z'\\in\\mathbb{R}^K\\).
Using \\(\\log p_i = z_i - \\mathrm{lse}(z)\\) where \\(\\mathrm{lse}(z)=\\log\\sum_k e^{z_k}\\),
\\[
\\boxed{~
d_H(p,q)
= \\max_i (z_i - z'_i) \\;-\\; \\min_i (z_i - z'_i)
\\;=\\; \\mathrm{range}(z - z') .
~}
\\]

*Proof sketch.* \\(\\log p_i - \\log q_i = (z_i - z'_i) - (\\mathrm{lse}(z)-\\mathrm{lse}(z'))\\); the additive constant cancels in \\(\\max-\\min\\).

**Consequence.** We can compute \\(d_H(p,q)\\) **without softmax**, via the **range of logit differences**.

### Projective Lipschitz constant

Define the global Lipschitz constant of \\(f_\\theta(x)=p_\\theta(x)\\) under \\(d_H\\):
\\[
\\mathrm{Lip}_H(f_\\theta)
:= \\sup_{x\\neq x'} \\frac{d_H\\big(p_\\theta(x),p_\\theta(x')\\big)}{\\|x-x'\\|}
= \\sup_{x\\neq x'} \\frac{\\mathrm{range}\\big(z_\\theta(x)-z_\\theta(x')\\big)}{\\|x-x'\\|}.
\\]

### Local differential view

Let \\(J_z(x)\\in\\mathbb{R}^{K\\times d}\\) be the Jacobian of logits. For unit \\(u\\),
\\[
\\limsup_{\\delta\\to 0} \\frac{d_H(p_\\theta(x+\\delta u),p_\\theta(x))}{\\delta}
= \\mathrm{range}\\big(J_z(x)\\,u\\big).
\\]
Equivalently (dual norm \\(\\|\\cdot\\|_*\\)),
\\[
\\sup_{i,j}\\|\\nabla_x z_i(x) - \\nabla_x z_j(x)\\|_*
\\quad\\text{is a local upper bound on}\\quad
\\mathrm{Lip}_H.
\\]

---

## Method: PLR penalties

PLR minimizes a batch-wise estimator of \\(\\mathrm{Lip}_H\\) using one of two lightweight variants.

### Variant A — Pairs *k*-NN (finite differences)

Within a mini-batch \\(B=\\{x_b\\}\\), form a pair set \\(\\mathcal{P}_B\\) using *k*-NN (on detached penultimate features \\(\\phi(x)\\) for stability). For \\((a,b)\\in\\mathcal{P}_B\\),
\\[
\\Delta z_{ab} := z_\\theta(x_a) - z_\\theta(x_b),
\\quad
\\delta_{ab} := \\|x_a - x_b\\|.
\\]
Define a hinge penalty with threshold \\(\\tau\\ge 0\\):
\\[
R_{\\mathrm{PLR}}^{\\mathrm{pairs}}(B)
= \\frac{1}{|\\mathcal{P}_B|} \\sum_{(a,b)\\in\\mathcal{P}_B}
\\bigg[ \\frac{\\mathrm{srange}_t(\\Delta z_{ab})}{\\delta_{ab}} - \\tau \\bigg]_+ .
\\]
Use \\(\\mathrm{srange}_t\\) (below) for differentiability.

**Complexity.** \\(O(k|B|K)\\) to evaluate \\(\\Delta z\\) and ranges; *k*-NN cost intra-batch.

### Variant B — Jacobian–vector proxy (Jv)

Avoids pairs and uses directional finite differences with unit vectors \\(v_r\\), \\(r=1..R\\) (e.g., random Gaussian normalized). For step \\(\\delta>0\\),
\\[
\\hat g(x;v_r) \\approx \\frac{z_\\theta(x+\\delta v_r) - z_\\theta(x-\\delta v_r)}{2\\delta}.
\\]
Penalty:
\\[
R_{\\mathrm{PLR}}^{\\mathrm{Jv}}(B)
= \\frac{1}{|B|R} \\sum_{x\\in B}\\sum_{r=1}^R
\\big[ \\mathrm{srange}_t\\big(\\hat g(x;v_r)\\big) - \\tau \\big]_+ .
\\]
**Complexity.** \\(O(R|B|)\\) Jacobian–vector products via autograd (no full Jacobian).

### Smoothed range (log-sum-exp)

The (non-smooth) range \\(\\mathrm{range}(v)=\\max_i v_i-\\min_i v_i\\) is replaced with a stable smooth approximation:
\\[
\\mathrm{smax}_t(v) := \\tfrac{1}{t}\\log\\!\\sum_i e^{t v_i},\\quad
\\mathrm{smin}_t(v) := -\\tfrac{1}{t}\\log\\!\\sum_i e^{-t v_i},\\quad
\\mathrm{srange}_t(v) := \\mathrm{smax}_t(v) - \\mathrm{smin}_t(v),
\\]
with temperature \\(t\\in[8,15]\\). As \\(t\\!\\to\\!\\infty\\), \\(\\mathrm{srange}_t \\to \\mathrm{range}\\).

### Full training objective

With cross-entropy \\(\\mathrm{CE}\\) and \\(\\lambda>0\\),
\\[
\\boxed{~
\\mathcal{L}(\\theta)
= \\mathbb{E}_{(x,y)}\\big[\\mathrm{CE}\\!\\big(p_\\theta(x),y\\big)\\big]
\\;+\\; \\lambda\\, \\mathbb{E}_{B}\\big[ R_{\\mathrm{PLR}}(B) \\big],
~}
\\]
where \\(R_{\\mathrm{PLR}}(B)\\) is either \\(R_{\\mathrm{PLR}}^{\\mathrm{pairs}}\\) or \\(R_{\\mathrm{PLR}}^{\\mathrm{Jv}}\\).

### Properties

- **Odds control.** For all \\(i,j\\),
  \\[
  \\big| \\log \\tfrac{p_i(x)}{p_j(x)} - \\log \\tfrac{p_i(x')}{p_j(x')} \\big|
  \\;\\le\\; d_H\\big(p(x),p(x')\\big).
  \\]
  PLR directly reduces this variation per input distance.
- **Shift invariance.** Invariant to \\(z\\mapsto z + c\\mathbf{1}\\) (projective geometry).
- **Birkhoff–Hopf intuition.** Contractions in Hilbert metric stabilize **odds**, hence **calibration** (we do not claim global contraction guarantees for arbitrary networks).

---

## Repository structure

```
plr/ 
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── configs/
│   ├── mnist_mlp.yaml
│   ├── mnist_mlp_plr.yaml
│   └── cifar10_smallcnn_plr.yaml
├── plr_hilbert/
│   ├── __init__.py
│   ├── train.py                # Entraînement (baseline / +PLR)
│   ├── eval.py                 # Éval + diagramme de fiabilité
│   ├── data/
│   │   └── datasets.py         # MNIST / CIFAR-10 (torchvision)
│   ├── models/
│   │   ├── mlp.py              # MLP simple
│   │   └── cnn_small.py        # Petit CNN pour CIFAR-10
│   ├── losses/
│   │   ├── plr.py              # ⚑ Pénalité PLR (Hilbert via range des logits)
│   │   └── calibration.py      # NLL, Brier, ECE
│   └── utils/
│       ├── seed.py             # Seeds + determinisme
│       ├── knn_pairs.py        # Paires k-NN intra-batch
│       └── logging_utils.py    # Sauvegarde config/metrics
├── scripts/
│   └── run_experiment.sh
└── tests/
    ├── test_hilbert_identity.py  # d_H(softmax(z), softmax(z')) == range(z - z')
    └── test_plr_backward.py      # test grad PLR (autograd)
```

---

## Installation

```bash
python3 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### Run commands

**MNIST — MLP (baseline):**
```bash
python -m plr_hilbert.train --config configs/mnist_mlp.yaml
```

**MNIST — MLP + PLR (pairs k-NN):**
```bash
python -m plr_hilbert.train --config configs/mnist_mlp_plr.yaml
```

**CIFAR-10 — small CNN + PLR:**
```bash
bash scripts/run_experiment.sh configs/cifar10_smallcnn_plr.yaml
```

**Evaluate & plot reliability:**
```bash
python -m plr_hilbert.eval --run runs/mnist_mlp_plr
```

### Example configuration

```yaml
# configs/mnist_mlp_plr.yaml
dataset: mnist
model:
  name: mlp
  hidden_dim: 256
train:
  epochs: 40
  batch_size: 128
  lr: 1e-3
  weight_decay: 1e-4
  seed: 42
plr:
  enabled: true
  variant: pairs         # "pairs" or "jv"
  lambda: 0.6            # λ
  tau: 0.30              # threshold τ
  k: 2                   # k-NN intra-batch (pairs)
  temp: 12.0             # srange temperature t
  use_penultimate: true  # k-NN on penultimate features (detach)
  warmup_epochs: 5
  jv:
    R: 3                 # num directions for Jv
    delta: 1e-3          # finite-diff step
log:
  out_dir: runs/mnist_mlp_plr
  save_every: 5
```

---

## Evaluation & calibration metrics

Implementations in `losses/calibration.py`:

- **NLL (Cross-Entropy)**:
  \\[
  \\mathrm{NLL} = -\\tfrac{1}{N} \\sum_{n=1}^N \\log p_\\theta(y_n \\mid x_n).
  \\]
- **Brier score**:
  \\[
  \\mathrm{Brier} = \\tfrac{1}{N}\\sum_{n=1}^N \\|p_\\theta(x_n) - e_{y_n}\\|_2^2.
  \\]
- **ECE (Expected Calibration Error)** with fixed bins \\(\\{I_b\\}\\):
  \\[
  \\mathrm{ECE} = \\sum_b \\frac{|I_b|}{N}\\; \\big|\\mathrm{acc}(I_b) - \\mathrm{conf}(I_b)\\big|.
  \\]
  (Optionally provide debiased ECE.)

`eval.py` logs CSV (Accuracy/NLL/Brier/ECE) and saves a **reliability diagram** PNG.

### Reliability diagram

For each confidence bin, plot predicted confidence vs empirical accuracy. A curve closer to the diagonal indicates better calibration. PLR should **reduce over-confidence** at similar accuracy.

---

## Hyperparameters & complexity

- **λ (strength)**: start with \\(0.3\\)–\\(0.8\\); use **warm-up** over 5–10 epochs.
- **τ (threshold)**: \\(0.25\\)–\\(0.35\\) ignores micro-variations; increase if the penalty dominates.
- **k (pairs)**: \\(2\\)–\\(4\\) neighbors suffice per sample (on **penultimate features** \\(\\phi(x)\\)).
- **t (temperature)**: \\(8\\)–\\(15\\) for \\(\\mathrm{srange}_t\\).
- **Jv**: \\(R\\in\\{2,3,4\\}\\), \\(\\delta \\approx 10^{-3}\\).

**Batch cost.**
- Pairs: \\(O(k|B|K)\\) + intra-batch k-NN (vectorized).
- Jv: \\(O(R|B|)\\) Jacobian–vector products (cheap via autograd).

**Notes.**
- Shift invariance avoids degenerate solutions tied to logit offsets.
- Prefer k-NN on detached \\(\\phi(x)\\) (penultimate) for semantic neighborhoods.

---

## Roadmap (paper-ready)

- **Datasets.** MNIST / F-MNIST (MLP), CIFAR-10 (small CNN), UCI Adult/Covertype (tabular).
- **Baselines.** CE only, **Label Smoothing**, **Temperature Scaling** (post-hoc), **Logit Pairing**, **Jacobian L2**.
- **Ablations.** \\(\\lambda,\\tau,k,t\\), warm-up; **pairs vs Jv**; k-NN on \\(\\phi(x)\\) vs input; effect of **penultimate layer choice**.
- **Local robustness.** Accuracy/NLL/ECE under light Gaussian noise and small-\\(\\varepsilon\\) FGSM.
- **Reporting.** Accuracy, NLL, Brier, ECE; reliability diagrams; batch-time overhead.

---

## Tests

Run:
```bash
pytest -q
```

- `tests/test_hilbert_identity.py`: checks
  \\[
  d_H(\\mathrm{softmax}(z), \\mathrm{softmax}(z')) \\approx \\mathrm{range}(z - z')
  \\]
  to float tolerance.
- `tests/test_plr_backward.py`: autograd gradient of PLR vs finite differences on logits.

---

## Citation

If you use this repository or the PLR idea, please cite:

```bibtex
@misc{plr_hilbert_2025,
  title   = {Hilbert-ML: Projective-Lipschitz Regularization for Calibration},
  author  = {Rehm, Rémi and collaborators},
  year    = {2025},
  url     = {https://github.com/<org>/plr-hilbert}
}
```

A `CITATION.cff` file is provided for scholarly metadata.

---

## License & acknowledgments

- See `LICENSE` for terms.  
- Contributions are welcome via issues/PRs.  
- PLR is a **simple, geometry-faithful** regularizer targeting the **probability simplex**; it complements Euclidean Jacobian/spectral penalties and post-hoc calibration.
