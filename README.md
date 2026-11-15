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

This repository provides the full reproducible code and controlled benchmarks for our study across **45 liquid U.S. equities and ETFs (2010–2025; 104,310 training and 44,910 out-of-sample windows)** alongside a **controlled synthetic state-transition and timescale benchmark ($H \in \{1, 5, 22, 60\}$ days)**.

### Key Findings

1. **On Empirical 5-Day Equity Realized Volatility, Untrained Random Features + TabICL Win:**
   - An **untrained Conv+Transformer encoder read out by TabICL** achieves the highest out-of-sample pooled $R^2$ ($0.501 \pm 0.008$ vs. $0.485$ for per-asset HAR-RV), the highest within-asset $R^2$ ($0.140 \pm 0.013$ vs. $0.131$), and the lowest log-MSE ($0.203 \pm 0.003$).
   - Pretraining the exact same encoder with a VICReg-style JEPA objective compresses the effective rank of the representations ($20.64 \to 13.85$) and degrades forecast accuracy. **Embedding-cluster batch sampling** partly restores both effective rank ($18.43$) and accuracy ($0.474 \pm 0.013$, Diebold–Mariano $t = -5.70$ and $-6.32$ vs. uniform batching).
   - **Pooled Panel $R^2$ Overstates Time-Series Skill:** Across all models, within-asset $R^2$ never exceeds $0.140$—meaning roughly three-quarters of pooled panel $R^2$ reflects persistent cross-sectional differences in baseline asset volatility rather than dynamic time-series forecasting.

2. **Controlled State-Machine & Timescale Phase Transition (Why JEPA Wins on State Transitions but Loses on Daily Equity Volatility):**
   - When the data-generating process is governed by a non-linear daily state-transition function $s_{t+1} = g(s_t, a_t)$ with high within-asset signal-to-noise ratio ($H = 1$ and $H = 5$), **the exact same JEPA + TabICL pipeline significantly beats the untrained encoder** ($t = -5.04$ to $-6.85$).
   - Conditioning the JEPA predictor on daily shocks $P_\theta(z_t, a_t)$ **linearizes the non-linear state transition in latent space**, raising a linear Ridge probe's within-asset state $R^2$ from $0.407$ (untrained) to $0.491$—virtually matching Untrained + TabICL ($0.494$).
   - **The Horizon / SNR Crossover:** As the forecasting horizon stretches ($H = 1 \to 5 \to 22 \to 60$) and unmeasured intervening shocks wash out predictable within-asset state dynamics ($\text{State } R^2: 0.524 \to 0.328 \to 0.005 \to -0.031$), the ranking reverses:
     - At **$H = 22$** (where within-asset RV $R^2 \approx 0.08$, matching empirical equities), **Uniform JEPA collapses** ($R^2 = -0.121$, $t = +5.50$) while **Cluster Batching prevents collapse** ($R^2 = +0.146$).
     - At **$H = 60$** (where only static cross-sectional scale remains), the **Untrained Encoder beats Cluster JEPA** ($t = +2.25$).

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

### 2. Controlled State-Transition & Timescale Sweep ($H \in \{1, 5, 22, 60\}$ Days)

| Horizon $H$ | Variant (Read Out by TabICL) | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | Latent State $R^2$ | DM vs. Untrained ($t$-stat) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$H = 1$** | A. Untrained Encoder | 19.00 | 0.154 | 0.141 | 0.485 | baseline |
| **$H = 1$** | C. JEPA Mag + Uniform | 10.08 | **0.180** | **0.167** | **0.524** | $-0.0290$ (**$-5.74$**) |
| **$H = 1$** | D. JEPA Mag + Cluster | 12.28 | 0.175 | 0.161 | 0.509 | $-0.0227$ (**$-5.05$**) |
| **$H = 1$** | E. Action-JEPA + Cluster | 13.01 | 0.174 | 0.161 | 0.482 | $-0.0224$ (**$-5.04$**) |
| **$H = 5$** | A. Untrained Encoder | 18.99 | 0.373 | 0.337 | 0.290 | baseline |
| **$H = 5$** | C. JEPA Mag + Uniform | 11.09 | 0.413 | 0.376 | 0.325 | $-0.0182$ (**$-6.17$**) |
| **$H = 5$** | D. JEPA Mag + Cluster | 11.49 | **0.414** | **0.377** | **0.328** | $-0.0186$ (**$-6.38$**) |
| **$H = 5$** | E. Action-JEPA + Cluster | 12.25 | 0.411 | 0.376 | 0.324 | $-0.0176$ (**$-6.85$**) |
| **$H = 22$** | A. Untrained Encoder | 18.95 | 0.137 | 0.076 | 0.001 | baseline |
| **$H = 22$** | C. JEPA Mag + Uniform | 13.30 | $-0.121$ | $-0.231$ | $-0.179$ | $+0.1022$ (**$+5.50$**) |
| **$H = 22$** | D. JEPA Mag + Cluster | 11.14 | 0.146 | **0.080** | 0.005 | $-0.0034$ ($-0.88$) |
| **$H = 22$** | E. Action-JEPA + Cluster | 11.94 | **0.148** | 0.078 | **0.014** | $-0.0044$ ($-0.89$) |
| **$H = 60$** | A. Untrained Encoder | 18.95 | 0.027 | **$-0.042$** | **$-0.024$** | baseline |
| **$H = 60$** | C. JEPA Mag + Uniform | 12.56 | $-0.259$ | $-0.463$ | $-0.176$ | $+0.0875$ (**$+3.57$**) |
| **$H = 60$** | D. JEPA Mag + Cluster | 12.93 | $-0.004$ | $-0.118$ | $-0.031$ | $+0.0096$ (**$+2.25$**) |
| **$H = 60$** | E. Action-JEPA + Cluster | 13.79 | **0.031** | $-0.091$ | $-0.049$ | $-0.0011$ ($-0.17$) |

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
    └── study3_state_timescale_experiments.py
```

---

## Quickstart & Installation

```bash
git clone https://github.com/DomArruda/jepa-tabicl-volatility.git
cd jepa-tabicl-volatility
pip install -r requirements.txt
python scripts/study3_state_timescale_experiments.py
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
