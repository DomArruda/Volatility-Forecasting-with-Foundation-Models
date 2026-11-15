# Random Features, Self-Supervised Latent Predictors, and Zero-Shot Tabular Foundation Models for Realized Volatility Forecasting

**A Controlled Study of Joint-Embedding Predictive Architectures (JEPA) Across State Transitions, Timescales, and Realized Volatility**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKH7SLEfa7P1ssgg7SL2jNicQjqRMGoG)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Author:** Dominic Arruda ([GitHub](https://github.com/DomArruda) | [LinkedIn](https://www.linkedin.com/in/dominic-arruda-15b9a0169/))  
**Date:** November 2026

---

## Overview

Can self-supervised latent world models—trained as **Joint-Embedding Predictive Architectures (JEPA)**—give zero-shot tabular foundation models (**TabICL**) better representations for financial time-series forecasting than an **untrained random initialization** of the exact same neural network?

This repository provides the full reproducible code and controlled benchmarks for our study across **45 liquid U.S. equities and ETFs (2010–2025; 104,310 training and 44,910 out-of-sample windows)** alongside a **controlled synthetic state-transition and timescale benchmark ($H \in \{1, 5, 22, 60\}$ days)** with a known ground-truth state machine $s_{t+1} = g(s_t, a_t) + \varepsilon$ — enabling oracle-scored measurement of state recovery, stability, and decision-making that is impossible on real data.

### Key Findings

1. **On Empirical 5-Day Equity Realized Volatility, Untrained Random Features + TabICL Win:**
   - An **untrained Conv+Transformer encoder read out by TabICL** achieves the highest out-of-sample pooled $R^2$ ($0.501 \pm 0.008$ vs. $0.485$ for per-asset HAR-RV), the highest within-asset $R^2$ ($0.140 \pm 0.013$ vs. $0.131$), and the lowest log-MSE ($0.203 \pm 0.003$).
   - Pretraining the exact same encoder with a VICReg-style JEPA objective compresses the effective rank of the representations ($20.64 \to 13.85$) and degrades forecast accuracy. **Embedding-cluster batch sampling** partly restores both effective rank ($18.43$) and accuracy ($0.474 \pm 0.013$, Diebold–Mariano $t = -5.70$ and $-6.32$ vs. uniform batching).
   - **Pooled Panel $R^2$ Overstates Time-Series Skill:** Across all models, within-asset $R^2$ never exceeds $0.140$—meaning roughly three-quarters of pooled panel $R^2$ reflects persistent cross-sectional differences in baseline asset volatility rather than dynamic time-series forecasting.

2. **Controlled State-Machine & Timescale Phase Transition (Why JEPA Wins on State Transitions but Loses on Daily Equity Volatility):** *(3 seeds: encoder init, JEPA training, and probe fits varied; panel fixed; mean ± sample std)*
   - When the data-generating process is governed by a non-linear daily state-transition function $s_{t+1} = g(s_t, a_t)$ with high within-asset signal-to-noise ratio ($H = 1$ and $H = 5$), **the exact same JEPA + TabICL pipeline significantly beats the untrained encoder** (DM $t = -4.1$ to $-5.9$).
   - Conditioning the JEPA predictor on daily shocks $P_\theta(z_t, a_t)$ **linearizes the non-linear state transition in latent space**, raising a linear Ridge probe's within-asset state $R^2$ from $0.424 \pm 0.016$ (untrained) to $0.492 \pm 0.003$—virtually matching Untrained + TabICL ($0.499 \pm 0.008$).
   - **The Horizon / SNR Crossover:** As the forecasting horizon stretches ($H = 1 \to 5 \to 22 \to 60$) and unmeasured intervening shocks wash out predictable within-asset state dynamics ($\text{State } R^2: 0.517 \to 0.324 \to 0.009 \to -0.035$), the ranking reverses:
     - At **$H = 22$** (where within-asset RV $R^2 \approx 0.08$, matching empirical equities), **Uniform JEPA collapses** ($R^2 = -0.130 \pm 0.007$, $t = +6.32 \pm 0.98$) while **Cluster Batching prevents collapse** ($R^2 = +0.146 \pm 0.013$, statistically tied with untrained, $t = -0.14 \pm 1.14$).
     - At **$H = 60$** (where only static cross-sectional scale remains), **Uniform JEPA collapses further** ($R^2 = -0.236 \pm 0.039$, $t = +5.82 \pm 2.82$) and the **Untrained Encoder is directionally ahead of Cluster JEPA** ($t = +1.74 \pm 0.96$, not significant across seeds) and best on all three mean metrics.
   - **Training buys stability, not just accuracy:** across seeds, JEPA-trained metrics are tight ($\pm 0.001$–$0.005$ in Experiment 1; Cluster effective rank pinned to $\pm 0.08$ at $H = 5$) while the untrained encoder is an initialization lottery ($\pm 0.015$ on Experiment-1 probes, $\pm 1.0$ effective rank), with its instability growing as horizon signal decays.

3. **Recovery ≠ Decision-Making (Belief-Space Planning on a Known State Machine):** *(Experiment C, H = 1, medium noise; oracle scored against the known $g$)*
   - We close the loop from probing to **control**: on a 27-point action grid, a 5-member ensemble of Action-JEPA predictors forms a *belief* over the next latent state (ensemble mean = expected dynamics, ensemble spread = estimated risk) and selects actions—scored against the oracle built from the true transition function.
   - **The representation that wins every probe loses the decision task.** Action-JEPA picks the exactly-optimal action *more often* (70.7% vs. 62.5% for untrained features) but its misses cost ~4× more (mean regret $0.217$ vs. $0.073$). Probing accuracy and decision quality **dissociate**: JEPA's predictor is non-linear in the action and extrapolates confidently-wrongly on off-distribution (counterfactual) action combinations, while a linear-in-action readout of random features degrades gracefully. Choosing a better action than history ever took is a counterfactual by definition—precisely the regime where the learned state machine is weakest.
   - **Belief-based safety filtering works from any representation:** filtering candidate actions by estimated $P(\text{dangerous log-vol state})$ and maximizing the conditional Sharpe $E[r']/\text{vol}'$ among survivors halves dangerous-state entries (42% → 22%) for only ~0.05 regret—for untrained features as well as JEPA.
   - **Risk is the price of return in this DGP:** the Sharpe-optimal action accepts a dangerous (90th-percentile log-vol) next state 42.8% of the time, because the regime action couples drift and volatility; the safety-first frontier is cheap.

---

## Summary of Results

### 1. Empirical Out-of-Sample Accuracy (45 U.S. Equities & ETFs, 2022–2025, 3 Seeds)

| Model Variant | Pooled $R^2$ | Within-Asset $R^2$ | IC (Pearson) | Log-MSE | Effective Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Untrained Encoder + TabICL** | **0.501 ± 0.008** | **0.140 ± 0.013** | **0.709** | **0.203 ± 0.003** | **20.64 ± 0.80** |
| HAR-RV log (per-asset OLS) | 0.485 | 0.131 | 0.707 | 0.209 | — |
| Raw Features + Ridge | 0.478 | 0.110 | 0.694 | 0.212 | — |
| **D. JEPA Mag + Cluster + TabICL** | 0.474 ± 0.013 | 0.085 ± 0.023 | 0.692 | 0.214 ± 0.005 | 18.43 ± 0.36 |
| HAR-RV log (pooled OLS) | 0.471 | 0.096 | 0.689 | 0.215 | — |
| Raw Features + TabICL | 0.470 ± 0.004 | 0.087 ± 0.006 | 0.690 | 0.215 ± 0.001 | — |
| A. Untrained Encoder + Ridge | 0.451 ± 0.009 | 0.067 ± 0.013 | 0.674 | 0.223 ± 0.004 | 20.64 ± 0.80 |
| C. JEPA Mag + Uniform + TabICL | 0.445 ± 0.015 | 0.038 ± 0.023 | 0.674 | 0.225 ± 0.006 | 13.85 ± 0.13 |
| B. JEPA LayerNorm + Uniform + TabICL | 0.444 ± 0.012 | 0.036 ± 0.019 | 0.674 | 0.226 ± 0.005 | 13.89 ± 0.45 |
| Naive ($\log \text{RV}_{22}$) | 0.403 | 0.020 | 0.684 | 0.243 | — |

---

### 2. Controlled State-Transition & Timescale Sweep ($H \in \{1, 5, 22, 60\}$ Days, 3 Seeds: Mean ± Std)

*Protocol: 3 seeds $\{42, 43, 44\}$ vary encoder init, JEPA training, and probe fits; panel (seed 2026) fixed; each seed's DM test uses that seed's untrained encoder. The earlier single-seed run (seed 42) reported $t = +2.25$ for Cluster at $H = 60$; across seeds this weakens to $+1.74 \pm 0.96$ (n.s.).*

| Horizon $H$ | Variant (Read Out by TabICL) | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | Latent State $R^2$ | DM $t$ vs. Untrained |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$H = 1$** | A. Untrained Encoder | 19.63 ± 0.98 | 0.156 ± 0.002 | 0.143 ± 0.002 | 0.495 ± 0.008 | baseline |
| **$H = 1$** | C. JEPA Mag + Uniform | 9.53 ± 0.49 | **0.177 ± 0.004** | **0.164 ± 0.003** | **0.517 ± 0.006** | **$-4.82 \pm 0.81$** |
| **$H = 1$** | D. JEPA Mag + Cluster | 11.29 ± 1.40 | 0.172 ± 0.003 | 0.160 ± 0.003 | 0.509 ± 0.008 | **$-4.08 \pm 0.32$** |
| **$H = 1$** | E. Action-JEPA + Cluster | 11.98 ± 0.99 | 0.173 ± 0.002 | 0.160 ± 0.002 | 0.498 ± 0.014 | **$-4.15 \pm 0.78$** |
| **$H = 5$** | A. Untrained Encoder | 19.62 ± 0.97 | 0.378 ± 0.005 | 0.342 ± 0.004 | 0.298 ± 0.007 | baseline |
| **$H = 5$** | C. JEPA Mag + Uniform | 11.52 ± 0.38 | 0.411 ± 0.003 | 0.374 ± 0.003 | 0.324 ± 0.001 | **$-4.85 \pm 1.20$** |
| **$H = 5$** | D. JEPA Mag + Cluster | 11.49 ± 0.08 | 0.411 ± 0.003 | 0.375 ± 0.002 | **0.324 ± 0.003** | **$-5.46 \pm 0.81$** |
| **$H = 5$** | E. Action-JEPA + Cluster | 12.22 ± 0.36 | **0.413 ± 0.004** | **0.378 ± 0.004** | 0.323 ± 0.002 | **$-5.87 \pm 1.08$** |
| **$H = 22$** | A. Untrained Encoder | 19.59 ± 0.97 | 0.145 ± 0.007 | **0.087 ± 0.009** | 0.006 ± 0.005 | baseline |
| **$H = 22$** | C. JEPA Mag + Uniform | 13.28 ± 0.27 | $-0.130 \pm 0.007$ | $-0.229 \pm 0.011$ | $-0.185 \pm 0.011$ | **$+6.32 \pm 0.98$** |
| **$H = 22$** | D. JEPA Mag + Cluster | 11.51 ± 0.38 | 0.146 ± 0.013 | 0.077 ± 0.012 | 0.003 ± 0.005 | $-0.14 \pm 1.14$ |
| **$H = 22$** | E. Action-JEPA + Cluster | 11.86 ± 0.08 | **0.151 ± 0.003** | 0.080 ± 0.001 | **0.009 ± 0.004** | $-0.45 \pm 0.42$ |
| **$H = 60$** | A. Untrained Encoder | 19.57 ± 0.95 | **0.031 ± 0.005** | **$-0.050 \pm 0.010$** | **$-0.019 \pm 0.008$** | baseline |
| **$H = 60$** | C. JEPA Mag + Uniform | 13.81 ± 1.19 | $-0.236 \pm 0.039$ | $-0.445 \pm 0.041$ | $-0.154 \pm 0.025$ | **$+5.82 \pm 2.82$** |
| **$H = 60$** | D. JEPA Mag + Cluster | 12.93 ± 0.20 | $-0.005 \pm 0.019$ | $-0.125 \pm 0.026$ | $-0.035 \pm 0.007$ | $+1.74 \pm 0.96$ |
| **$H = 60$** | E. Action-JEPA + Cluster | 13.79 ± 0.05 | 0.003 ± 0.026 | $-0.121 \pm 0.030$ | $-0.046 \pm 0.007$ | $+0.85 \pm 0.90$ |

---

### 3. Belief-Based Action Selection ($H = 1$, Medium Noise, 27-Action Grid, 5-Member Ensembles)

*Utility = conditional Sharpe $E[r']/\text{vol}'$ (per the panel construction $E[r_{t+1}] = 0.0005\,m'$, $\text{vol} = e^{l'}/\sqrt{252}$); danger = next log-vol level above the 90th percentile of panel levels ($\tau = -2.105$); regret scored against the oracle built from the true transition function.*

| Policy | Mean True Utility | Mean Regret | % Optimal | % Dangerous |
| :--- | :---: | :---: | :---: | :---: |
| Oracle (true $g$) | −0.645 | 0.000 | 100.0 | 42.8 |
| Action-JEPA Ens. risk-neutral | −0.862 | 0.217 | **70.7** | 42.0 |
| Action-JEPA Ens. safety-first | −0.908 | 0.263 | 33.6 | 22.2 |
| Untrained Ens. risk-neutral | −0.718 | **0.073** | 62.5 | 42.2 |
| Untrained Ens. safety-first | −0.770 | 0.125 | 27.8 | 23.5 |
| Random action | −1.290 | 0.644 | 3.0 | 13.9 |
| Zero action (do nothing) | −1.459 | 0.814 | 0.0 | 6.6 |

---

## Repository Structure

```text
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── paper/
│   ├── Arruda_RandomFeatures_JEPA_TabICL_Volatility_v3.md
│   └── Arruda_RandomFeatures_JEPA_TabICL_Volatility_v3.docx
└── scripts/
    ├── study3_state_timescale_experiments.py   # Experiments 1–2: state-transition recovery + H×{1,5,22,60} timescale sweep (3 seeds)
    └── study3_ablations.py                     # Ablations A/B (random-linear baseline; noise×horizon×order recovery),
                                                # Experiment C (belief-based action selection), Experiment D (portfolio
                                                # allocation vs. equal-weight/inverse-vol/oracle), Experiment E (LP
                                                # decision-complexity scaling) — self-contained, Colab-runnable
```

---

## Quickstart & Installation

```bash
git clone https://github.com/DomArruda/Volatility-Forecasting-with-Foundational-Models.git
cd jepa-tabicl-volatility
pip install -r requirements.txt
python scripts/study3_state_timescale_experiments.py
python scripts/study3_ablations.py   # Ablations A/B + Experiments C/D/E (self-contained, Colab-runnable)
```

---

## Citation

```bibtex
@techreport{arruda2026random,
  title={Random Features, Self-Supervised Latent Predictors, and Zero-Shot Tabular Foundation Models for Realized Volatility Forecasting},
  author={Arruda, Dominic},
  year={2026},
  month={November},
  institution={Bryant University and Providence College},
  note={Working Paper (v3)}
}
```

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
