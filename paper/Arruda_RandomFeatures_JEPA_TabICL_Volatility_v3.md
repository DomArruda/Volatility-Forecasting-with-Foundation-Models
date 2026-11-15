# Random Features, Self-Supervised Latent Predictors, and Zero-Shot Tabular Foundation Models for Realized Volatility Forecasting
**A Controlled Study of Joint-Embedding Predictive Architectures (JEPA) Across State Transitions, Timescales, and Realized Volatility**

**Dominic Arruda** | November 2026  
**Keywords:** realized volatility; self-supervised learning; JEPA; random features; tabular foundation models; in-context learning; state-space models; HAR-RV; forecast evaluation

---

## Abstract
We ask whether self-supervised latent world models, trained as Joint-Embedding Predictive Architectures (JEPA), give zero-shot tabular foundation models better inputs for forecasting 5-day realized volatility. On 45 liquid U.S. equities and ETFs (daily data, 2010–2025; 104,310 training and 44,910 out-of-sample windows from 2022–2025), we compare three pretrained JEPA variants (LayerNorm bottleneck; magnitude-preserving bottleneck; magnitude-preserving with embedding-cluster batch sampling) against an untrained copy of the same encoder, raw features, HAR-RV and a naive benchmark, each read out by ridge regression and by TabICL, over three seeds.

The untrained encoder read out by TabICL has the highest out-of-sample $R^2$ on log realized volatility ($0.501 \pm 0.008$ vs. $0.485$ for per-asset HAR-RV) and the lowest log-MSE ($0.203$). Every pretrained JEPA variant is worse on average than its untrained counterpart, and the best of them is significantly worse in two of three seeds. The effective rank of the embeddings tracks forecast accuracy on empirical equities (untrained $20.6$; cluster-batched JEPA $18.4$; uniform-batched JEPA $13.9$), and embedding-cluster batching recovers part of the loss. No model explains more than 14% of within-asset variation, so most pooled $R^2$ reflects persistent differences in volatility levels across assets.

To verify that this reversal is not an implementation artifact, we run a controlled synthetic experiment testing whether the exact same JEPA + TabICL architecture can replicate non-linear daily state-transition functions $s_{t+1} = g(s_t, a_t)$ across forecasting horizons $H \in \{1, 5, 22, 60\}$ (three seeds; mean ± sample standard deviation). When volatility is driven by predictable state transitions ($H = 1$ and $H = 5$), every JEPA variant significantly beats the untrained encoder (DM $t = -4.1$ to $-5.9$), and an action-conditioned predictor $P_\theta(z_t, a_t)$ linearizes the transition operator so that a linear ridge probe ($R^2_{\text{within}} = 0.492 \pm 0.003$) nearly matches the untrained encoder with TabICL ($0.499 \pm 0.008$). However, as horizon decay ($H = 22$ and $H = 60$) pushes within-asset predictability below $0.08$—matching the low signal-to-noise ratio of daily equities—uniform-batched JEPA collapses ($R^2 = -0.130 \pm 0.007$, $t = +6.32 \pm 0.98$), cluster batching prevents the collapse ($R^2 = +0.146 \pm 0.013$, statistically tied with the untrained encoder), and at $H = 60$ the untrained encoder is directionally ahead of cluster JEPA ($t = +1.74 \pm 0.96$, not significant across seeds) and best on all three mean metrics.

---

## Section 6.2. Controlled Experiment: State-Transition Replication and Timescale Decay

A natural objection to our empirical findings (F1 and F2) is that the 32-width encoder, the VICReg objective, or the TabICL readout may simply be misconfigured, preventing JEPA from learning useful time-series representations under any data-generating process. Conversely, if the architecture is sound, the empirical reversal should be reproducible in a synthetic panel as a function of the within-asset signal-to-noise ratio (SNR) across forecasting horizons. We run two controlled experiments on a synthetic 20-asset panel ($T = 1,400$ days, $40 \times 8$ context windows standardized on the first 1,000 days, H-day embargo split) constructed with the exact eight backward-looking features, encoder architecture, EMA target encoder ($\tau = 0.996$), VICReg weights (12, 10, 1), cluster sampler ($K = 16$, 85% within-cluster), and TabICL/ridge probes used in Study 3. All synthetic results are reported as mean ± sample standard deviation over three seeds (encoder initialization, JEPA training, and probe fits varied; panel seed fixed at 2026), with each seed's Diebold–Mariano test using that seed's untrained encoder as baseline; the first seed reproduces the original single-run protocol.

Each asset $i$ has a persistent baseline log-volatility $\beta_i \sim \text{Uniform}(-2.2, -1.0)$ and a three-dimensional latent state $s_{i,t} = [\ell_{i,t}, \mu_{i,t}, \rho_{i,t}]^\top$ governing log-volatility level, drift regime, and persistence. On each day $t$, a sparse functional intervention $a_{i,t} = [a_{\text{shock}}, a_{\text{regime}}, a_{\text{damp}}]^\top \in \mathbb{R}^3$ (non-zero with probability 0.25) moves the asset to a new state via a non-linear transition operator $s_{i,t+1} = g(s_{i,t}, a_{i,t}; \beta_i) + \eta_{i,t}$.

### Table 5. Experiment 1: Replicating the Non-Linear State Transition $g(s_t, a_t)$, Out-of-Sample (3 Seeds: Mean ± Std)

| Model Variant | Pooled State $R^2$ | Within-Asset State $R^2$ | Transition MSE |
| :--- | :---: | :---: | :---: |
| A. Untrained encoder $[z_t, a_t]$ + Ridge | 0.454 ± 0.015 | 0.424 ± 0.016 | 1.957 ± 0.034 |
| A. Untrained encoder $[z_t, a_t]$ + TabICL | 0.527 ± 0.007 | 0.499 ± 0.008 | 1.664 ± 0.012 |
| D. JEPA Mag + Cluster $[z_t, a_t]$ + TabICL | **0.558 ± 0.001** | **0.533 ± 0.002** | **1.553 ± 0.004** |
| E. Action-JEPA $P(z_t, a_t)$ + Ridge | 0.520 ± 0.003 | 0.492 ± 0.003 | 1.657 ± 0.007 |
| E. Action-JEPA $[P(z_t, a_t), a_t]$ + TabICL | 0.553 ± 0.003 | 0.528 ± 0.003 | 1.573 ± 0.009 |

### Table 6. Experiment 2: Representation Rank and Forecast Accuracy Across Horizons $H \in \{1, 5, 22, 60\}$ (3 Seeds: Mean ± Std)

| Horizon $H$ | Variant (read out by TabICL) | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | State $H$ $R^2$ | DM $t$ vs. Untrained |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $H = 1$ | A. Untrained encoder | 19.63 ± 0.98 | 0.156 ± 0.002 | 0.143 ± 0.002 | 0.495 ± 0.008 | baseline |
| $H = 1$ | C. JEPA Mag + Uniform | 9.53 ± 0.49 | **0.177 ± 0.004** | **0.164 ± 0.003** | **0.517 ± 0.006** | **$-4.82 \pm 0.81$** |
| $H = 1$ | D. JEPA Mag + Cluster | 11.29 ± 1.40 | 0.172 ± 0.003 | 0.160 ± 0.003 | 0.509 ± 0.008 | **$-4.08 \pm 0.32$** |
| $H = 1$ | E. Action-JEPA + Cluster | 11.98 ± 0.99 | 0.173 ± 0.002 | 0.160 ± 0.002 | 0.498 ± 0.014 | **$-4.15 \pm 0.78$** |
| $H = 5$ | A. Untrained encoder | 19.62 ± 0.97 | 0.378 ± 0.005 | 0.342 ± 0.004 | 0.298 ± 0.007 | baseline |
| $H = 5$ | C. JEPA Mag + Uniform | 11.52 ± 0.38 | 0.411 ± 0.003 | 0.374 ± 0.003 | 0.324 ± 0.001 | **$-4.85 \pm 1.20$** |
| $H = 5$ | D. JEPA Mag + Cluster | 11.49 ± 0.08 | 0.411 ± 0.003 | 0.375 ± 0.002 | **0.324 ± 0.003** | **$-5.46 \pm 0.81$** |
| $H = 5$ | E. Action-JEPA + Cluster | 12.22 ± 0.36 | **0.413 ± 0.004** | **0.378 ± 0.004** | 0.323 ± 0.002 | **$-5.87 \pm 1.08$** |
| $H = 22$ | A. Untrained encoder | 19.59 ± 0.97 | 0.145 ± 0.007 | **0.087 ± 0.009** | 0.006 ± 0.005 | baseline |
| $H = 22$ | C. JEPA Mag + Uniform | 13.28 ± 0.27 | $-0.130 \pm 0.007$ | $-0.229 \pm 0.011$ | $-0.185 \pm 0.011$ | **$+6.32 \pm 0.98$** |
| $H = 22$ | D. JEPA Mag + Cluster | 11.51 ± 0.38 | 0.146 ± 0.013 | 0.077 ± 0.012 | 0.003 ± 0.005 | $-0.14 \pm 1.14$ |
| $H = 22$ | E. Action-JEPA + Cluster | 11.86 ± 0.08 | **0.151 ± 0.003** | 0.080 ± 0.001 | **0.009 ± 0.004** | $-0.45 \pm 0.42$ |
| $H = 60$ | A. Untrained encoder | 19.57 ± 0.95 | **0.031 ± 0.005** | **$-0.050 \pm 0.010$** | **$-0.019 \pm 0.008$** | baseline |
| $H = 60$ | C. JEPA Mag + Uniform | 13.81 ± 1.19 | $-0.236 \pm 0.039$ | $-0.445 \pm 0.041$ | $-0.154 \pm 0.025$ | **$+5.82 \pm 2.82$** |
| $H = 60$ | D. JEPA Mag + Cluster | 12.93 ± 0.20 | $-0.005 \pm 0.019$ | $-0.125 \pm 0.026$ | $-0.035 \pm 0.007$ | $+1.74 \pm 0.96$ |
| $H = 60$ | E. Action-JEPA + Cluster | 13.79 ± 0.05 | 0.003 ± 0.026 | $-0.121 \pm 0.030$ | $-0.046 \pm 0.007$ | $+0.85 \pm 0.90$ |

### Findings from Controlled Experiments
- **F7. JEPA outperforms random features when latent state transitions are predictable, and linearizes the transition operator.** When the evaluation target is the true post-intervention state $g(s_t, a_t)$ (Table 5) or short-horizon realized volatility driven by a persistent state ($H = 1$ and $H = 5$ in Table 6), every pretrained JEPA variant significantly beats the untrained encoder (DM $t = -4.1$ to $-5.9$, mean ± s.d. across three seeds). Conditioning the predictor on the daily intervention (Variant E) linearizes the non-linear transition inside the embedding space: a linear ridge probe on $P_\theta(z_t, a_t)$ achieves a within-asset $R^2$ of $0.492 \pm 0.003$, gaining $+6.8$ percentage points over the untrained encoder with ridge ($0.424 \pm 0.016$) and nearly matching the untrained encoder with TabICL ($0.499 \pm 0.008$).
- **F8. Rank compression denoises when state SNR is high, but hurts when only static scale remains.** At $H = 1$ and $H = 5$, compressing the 33-dimensional embedding down to an effective rank of $9.5\text{–}12.2$ projects out orthogonal return noise around the three-dimensional true state and improves TabICL accuracy. Once horizon decay ($H \ge 22$) pushes within-asset RV $R^2$ below $0.08$, Uniform JEPA collapses ($R^2 = -0.130 \pm 0.007$, $t = +6.32 \pm 0.98$), Cluster Batching prevents collapse ($R^2 = +0.146 \pm 0.013$, statistically tied with the untrained encoder, $t = -0.14 \pm 1.14$), and by $H = 60$ the untrained encoder is significantly ahead of Uniform JEPA ($t = +5.82 \pm 2.82$) and directionally ahead of Cluster JEPA ($t = +1.74 \pm 0.96$, not significant across seeds).
- **F9. Empirical daily equity volatility sits in the low-state-SNR regime.** In daily close-to-close equity data without exogenous shock covariates, within-asset time-series predictability is low ($R^2 \le 0.14$) and roughly three-quarters of predictable variance is static cross-sectional scale (F4), placing empirical equities in the $H \ge 22$ regime of Table 6.
