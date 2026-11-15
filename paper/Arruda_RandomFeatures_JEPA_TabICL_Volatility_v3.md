# Random Features, Self-Supervised Latent Predictors, and Zero-Shot Tabular Foundation Models for Realized Volatility Forecasting
**A Controlled Study of Joint-Embedding Predictive Architectures (JEPA) Across State Transitions, Timescales, and Realized Volatility**

**Dominic Arruda** | November 2026  
**Keywords:** realized volatility; self-supervised learning; JEPA; random features; tabular foundation models; in-context learning; state-space models; HAR-RV; forecast evaluation

---

## Abstract
We ask whether self-supervised latent world models, trained as Joint-Embedding Predictive Architectures (JEPA), give zero-shot tabular foundation models better inputs for forecasting 5-day realized volatility. On 45 liquid U.S. equities and ETFs (daily data, 2010–2025; 104,310 training and 44,910 out-of-sample windows from 2022–2025), we compare three pretrained JEPA variants (LayerNorm bottleneck; magnitude-preserving bottleneck; magnitude-preserving with embedding-cluster batch sampling) against an untrained copy of the same encoder, raw features, HAR-RV and a naive benchmark, each read out by ridge regression and by TabICL, over three seeds.

The untrained encoder read out by TabICL has the highest out-of-sample $R^2$ on log realized volatility ($0.501 \pm 0.008$ vs. $0.485$ for per-asset HAR-RV) and the lowest log-MSE ($0.203$). Every pretrained JEPA variant is worse on average than its untrained counterpart, and the best of them is significantly worse in two of three seeds. The effective rank of the embeddings tracks forecast accuracy on empirical equities (untrained $20.6$; cluster-batched JEPA $18.4$; uniform-batched JEPA $13.9$), and embedding-cluster batching recovers part of the loss. No model explains more than 14% of within-asset variation, so most pooled $R^2$ reflects persistent differences in volatility levels across assets.

To verify that this reversal is not an implementation artifact, we run a controlled synthetic experiment testing whether the exact same JEPA + TabICL architecture can replicate non-linear daily state-transition functions $s_{t+1} = g(s_t, a_t)$ across forecasting horizons $H \in \{1, 5, 22, 60\}$. When volatility is driven by predictable state transitions ($H = 1$ and $H = 5$), every JEPA variant significantly beats the untrained encoder ($t = -5.04$ to $-6.85$), and an action-conditioned predictor $P_\theta(z_t, a_t)$ linearizes the transition operator so that a linear ridge probe ($R^2_{\text{within}} = 0.491$) nearly matches the untrained encoder with TabICL ($0.494$). However, as horizon decay ($H = 22$ and $H = 60$) pushes within-asset predictability below $0.08$—matching the low signal-to-noise ratio of daily equities—uniform-batched JEPA collapses ($R^2 = -0.121$, $t = +5.50$), cluster batching prevents the collapse ($R^2 = +0.146$), and the untrained encoder overtakes cluster JEPA ($t = +2.25$).

---

## Section 6.2. Controlled Experiment: State-Transition Replication and Timescale Decay

A natural objection to our empirical findings (F1 and F2) is that the 32-width encoder, the VICReg objective, or the TabICL readout may simply be misconfigured, preventing JEPA from learning useful time-series representations under any data-generating process. Conversely, if the architecture is sound, the empirical reversal should be reproducible in a synthetic panel as a function of the within-asset signal-to-noise ratio (SNR) across forecasting horizons. We run two controlled experiments on a synthetic 20-asset panel ($T = 1,400$ days, $40 \times 8$ context windows standardized on the first 1,000 days, H-day embargo split) constructed with the exact eight backward-looking features, encoder architecture, EMA target encoder ($\tau = 0.996$), VICReg weights (12, 10, 1), cluster sampler ($K = 16$, 85% within-cluster), and TabICL/ridge probes used in Study 3.

Each asset $i$ has a persistent baseline log-volatility $\beta_i \sim \text{Uniform}(-2.2, -1.0)$ and a three-dimensional latent state $s_{i,t} = [\ell_{i,t}, \mu_{i,t}, \rho_{i,t}]^\top$ governing log-volatility level, drift regime, and persistence. On each day $t$, a sparse functional intervention $a_{i,t} = [a_{\text{shock}}, a_{\text{regime}}, a_{\text{damp}}]^\top \in \mathbb{R}^3$ (non-zero with probability 0.25) moves the asset to a new state via a non-linear transition operator $s_{i,t+1} = g(s_{i,t}, a_{i,t}; \beta_i) + \eta_{i,t}$.

### Table 5. Experiment 1: Replicating the Non-Linear State Transition $g(s_t, a_t)$, Out-of-Sample

| Model Variant | Pooled State $R^2$ | Within-Asset State $R^2$ | Transition MSE |
| :--- | :---: | :---: | :---: |
| A. Untrained encoder $[z_t, a_t]$ + Ridge | 0.438 | 0.407 | 1.991 |
| A. Untrained encoder $[z_t, a_t]$ + TabICL | 0.523 | 0.494 | 1.661 |
| D. JEPA Mag + Cluster $[z_t, a_t]$ + TabICL | **0.556** | **0.531** | **1.551** |
| E. Action-JEPA $P(z_t, a_t)$ + Ridge | 0.520 | 0.491 | 1.665 |
| E. Action-JEPA $[P(z_t, a_t), a_t]$ + TabICL | 0.551 | 0.525 | 1.579 |

### Table 6. Experiment 2: Representation Rank and Forecast Accuracy Across Horizons $H \in \{1, 5, 22, 60\}$

| Horizon $H$ | Variant (read out by TabICL) | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | State $H$ $R^2$ | DM vs. Untrained ($t$-stat) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $H = 1$ | A. Untrained encoder | 19.00 | 0.154 | 0.141 | 0.485 | baseline |
| $H = 1$ | C. JEPA Mag + Uniform | 10.08 | **0.180** | **0.167** | **0.524** | $-0.0290$ ($-5.74$) |
| $H = 1$ | D. JEPA Mag + Cluster | 12.28 | 0.175 | 0.161 | 0.509 | $-0.0227$ ($-5.05$) |
| $H = 1$ | E. Action-JEPA + Cluster | 13.01 | 0.174 | 0.161 | 0.482 | $-0.0224$ ($-5.04$) |
| $H = 5$ | A. Untrained encoder | 18.99 | 0.373 | 0.337 | 0.290 | baseline |
| $H = 5$ | C. JEPA Mag + Uniform | 11.09 | 0.413 | 0.376 | 0.325 | $-0.0182$ ($-6.17$) |
| $H = 5$ | D. JEPA Mag + Cluster | 11.49 | **0.414** | **0.377** | **0.328** | $-0.0186$ ($-6.38$) |
| $H = 5$ | E. Action-JEPA + Cluster | 12.25 | 0.411 | 0.376 | 0.324 | $-0.0176$ ($-6.85$) |
| $H = 22$ | A. Untrained encoder | 18.95 | 0.137 | 0.076 | 0.001 | baseline |
| $H = 22$ | C. JEPA Mag + Uniform | 13.30 | $-0.121$ | $-0.231$ | $-0.179$ | $+0.1022$ ($+5.50$) |
| $H = 22$ | D. JEPA Mag + Cluster | 11.14 | 0.146 | **0.080** | 0.005 | $-0.0034$ ($-0.88$) |
| $H = 22$ | E. Action-JEPA + Cluster | 11.94 | **0.148** | 0.078 | **0.014** | $-0.0044$ ($-0.89$) |
| $H = 60$ | A. Untrained encoder | 18.95 | 0.027 | **$-0.042$** | **$-0.024$** | baseline |
| $H = 60$ | C. JEPA Mag + Uniform | 12.56 | $-0.259$ | $-0.463$ | $-0.176$ | $+0.0875$ ($+3.57$) |
| $H = 60$ | D. JEPA Mag + Cluster | 12.93 | $-0.004$ | $-0.118$ | $-0.031$ | $+0.0096$ ($+2.25$) |
| $H = 60$ | E. Action-JEPA + Cluster | 13.79 | **0.031** | $-0.091$ | $-0.049$ | $-0.0011$ ($-0.17$) |

### Findings from Controlled Experiments
- **F7. JEPA outperforms random features when latent state transitions are predictable, and linearizes the transition operator.** When the evaluation target is the true post-intervention state $g(s_t, a_t)$ (Table 5) or short-horizon realized volatility driven by a persistent state ($H = 1$ and $H = 5$ in Table 6), every pretrained JEPA variant significantly beats the untrained encoder ($t = -5.04$ to $-6.85$). Conditioning the predictor on the daily intervention (Variant E) linearizes the non-linear transition inside the embedding space: a linear ridge probe on $P_\theta(z_t, a_t)$ achieves a within-asset $R^2$ of $0.491$, gaining $+8.4$ percentage points over the untrained encoder with ridge ($0.407$) and nearly matching the untrained encoder with TabICL ($0.494$).
- **F8. Rank compression denoises when state SNR is high, but hurts when only static scale remains.** At $H = 1$ and $H = 5$, compressing the 33-dimensional embedding down to an effective rank of $10.1\text{–}13.0$ projects out orthogonal return noise around the three-dimensional true state and improves TabICL accuracy. Once horizon decay ($H \ge 22$) pushes within-asset RV $R^2$ below $0.08$, Uniform JEPA collapses ($R^2 = -0.121$, $t = +5.50$), Cluster Batching prevents collapse ($R^2 = +0.146$), and by $H = 60$ the untrained encoder beats both Uniform JEPA ($t = +3.57$) and Cluster JEPA ($t = +2.25$).
- **F9. Empirical daily equity volatility sits in the low-state-SNR regime.** In daily close-to-close equity data without exogenous shock covariates, within-asset time-series predictability is low ($R^2 \le 0.14$) and roughly three-quarters of predictable variance is static cross-sectional scale (F4), placing empirical equities in the $H \ge 22$ regime of Table 6.
