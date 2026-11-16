# Random Features, Self-Supervised Latent Predictors, and Zero-Shot Tabular Foundation Models for Realized Volatility Forecasting

**A Controlled Study of Joint-Embedding Predictive Architectures (JEPA) Across State Transitions, Timescales, and Realized Volatility**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Author:** Dominic Arruda ([GitHub](https://github.com/DomArruda) | LinkedIn <!-- add LinkedIn URL -->)
**Date:** November 2026

## Overview

Can self-supervised latent world models, trained as Joint-Embedding Predictive Architectures (JEPA), give zero-shot tabular foundation models (TabICL) better representations for financial time-series forecasting than an untrained random initialization of the exact same network?

This repository contains the code and controlled benchmarks for the study:

- **Empirical:** 45 liquid U.S. equities and ETFs, 2010–2025 (104,310 training and 44,910 out-of-sample windows; out-of-sample period 2022–2025).
- **Synthetic:** a state-transition and timescale benchmark ($H \in \{1, 5, 22, 60\}$ days) with a known ground-truth state machine $s_{t+1} = g(s_t, a_t) + \varepsilon$. Because $g$ is known, state recovery, stability, and decisions can be scored against an oracle, which is impossible on real data.

## Key Findings

### Empirical 5-day equity realized volatility: untrained random features + TabICL do best

- An untrained Conv+Transformer encoder read out by TabICL has the highest out-of-sample pooled $R^2$ ($0.501 \pm 0.008$ vs. $0.485$ for per-asset HAR-RV), the highest within-asset $R^2$ ($0.140 \pm 0.013$ vs. $0.131$), and the lowest log-MSE ($0.203 \pm 0.003$). The within-asset margin over per-asset HAR-RV is inside the seed standard deviation, so the safe reading is that random features + TabICL **match or slightly exceed** the strongest classical baseline. A Diebold–Mariano test against per-asset HAR-RV is the next check.
- Pretraining the same encoder with a VICReg-style JEPA objective lowers the effective rank of its representations ($20.64 \to 13.85$) and lowers forecast accuracy. Embedding-cluster batch sampling partly restores both (effective rank $18.43$; pooled $R^2$ $0.474 \pm 0.013$; Diebold–Mariano $t = -5.70$ and $-6.32$ vs. the two uniform-batching variants <!-- TODO: say which t is vs. B and which vs. C -->). Ablation A shows that effective rank alone does **not** predict quality, so the rank drop is best read as a symptom, not the cause.
- **Pooled panel $R^2$ overstates time-series skill.** Across all models, within-asset $R^2$ never exceeds $0.140$. As a rough heuristic ($1 - 0.140/0.501 \approx 72\%$; $R^2$ does not decompose exactly), about three-quarters of pooled $R^2$ reflects persistent cross-sectional differences in baseline volatility rather than time-series forecasting.

### Controlled state machine and timescale sweep: why JEPA wins on state transitions but loses on daily equity volatility

*3 seeds: encoder init, JEPA training, and probe fits varied; panel fixed; mean ± sample std.*

- When the data-generating process is a non-linear daily state transition $s_{t+1} = g(s_t, a_t)$ with high within-asset signal-to-noise ($H = 1$ and $H = 5$), the same JEPA + TabICL pipeline significantly beats the untrained encoder (DM $t = -4.1$ to $-5.9$).
- Conditioning the JEPA predictor on daily shocks, $P_\theta(z_t, a_t)$, linearizes the state transition in latent space: in Experiment 1, a linear Ridge probe's within-asset state $R^2$ rises from $0.424 \pm 0.016$ (untrained) to $0.492 \pm 0.003$, nearly matching Untrained + TabICL ($0.499 \pm 0.008$).
- **The horizon / SNR crossover.** As the horizon stretches ($H = 1 \to 5 \to 22 \to 60$), unmeasured intervening shocks wash out the predictable within-asset state dynamics (Uniform JEPA state $R^2$: $0.517 \to 0.324 \to -0.185 \to -0.154$; untrained: $0.495 \to 0.298 \to 0.006 \to -0.019$), and the ranking reverses:
  - At $H = 22$ (within-asset RV $R^2 \approx 0.08$, similar to empirical equities), Uniform JEPA collapses ($R^2 = -0.130 \pm 0.007$, $t = +6.32 \pm 0.98$), while Cluster Batching prevents the collapse ($R^2 = +0.146 \pm 0.013$, statistically tied with untrained, $t = -0.14 \pm 1.14$).
  - At $H = 60$ (only static cross-sectional scale remains), Uniform JEPA collapses further ($R^2 = -0.236 \pm 0.039$, $t = +5.82 \pm 2.82$). The untrained encoder is directionally ahead of Cluster JEPA ($t = +1.74 \pm 0.96$, not significant across 3 seeds) and best on all three mean metrics.
- **Stability depends on horizon.** At $H \le 5$, JEPA-trained metrics are tight across seeds (e.g., Cluster effective rank $\pm 0.08$ at $H = 5$; RV $R^2$ $\pm 0.002$–$0.004$), while the untrained encoder varies with initialization ($\pm 1.0$ effective rank). This does not hold everywhere: at $H = 1$ Cluster's effective rank varies *more* than the untrained encoder's ($\pm 1.40$ vs. $\pm 0.98$), and at $H = 60$ Uniform JEPA's pooled $R^2$ spread ($\pm 0.039$) is about 8× the untrained encoder's ($\pm 0.005$). Training buys stability only while there is horizon signal to learn.

### Ablations (seed 42, original DGP)

- **The untrained architecture matters.** A frozen random linear projection of the same width (33 dims) recovers less than the untrained Conv+Transformer at $H = 5$, with either readout: state $R^2$ $0.177$ vs. $0.253$ with Ridge, and $0.236$ vs. $0.290$ with TabICL. The random-features result is not "any random projection works."
- **TabICL beats Ridge on every extractor** (RV pooled $R^2$ +0.025 to +0.050), and JEPA Cluster + TabICL is best ($0.414$ pooled, $0.329$ state), consistent with Experiment 2 at $H = 5$.
- **Effective rank does not predict quality.** The random linear projection has the *highest* effective rank (29.3) and the worst accuracy; JEPA Cluster has the lowest (11.4) and the best.
- **Horizon dominates process noise.** Moving from low to high process noise lowers $H = 1$ first-order recovery only from $0.472$ to $0.445$ (JEPA Cluster), while moving from $H = 1$ to $H = 22$ takes every extractor to about zero. The unmeasured action shocks, not the state noise, are the main source of unpredictability.

### Decision experiments: revised in v3.1

The v3 decision experiments (C–G) used a DGP in which the drift state is **negative in 100% of states** (mean $m = -3.04$), because the drift equation couples to the raw log-vol level, $\tanh(a_{\text{regime}} + 0.3\,l)$, with $l \approx -4.7$. With negative drift, the conditional-Sharpe utility $m'/\text{vol}'$ is maximized by **raising** volatility. The v3 finding that "the Sharpe-optimal action accepts a dangerous state 42.8% of the time because the regime action couples drift and volatility" was this artifact. The same DGP also leaves 40% of states below the volatility clip floor, where returns carry no information about $l$.

v3.1 runs C–G on a **centered** DGP that changes one term, $\tanh(a_{\text{regime}} + 0.3\,(l - \beta))$, coupling drift to each asset's own baseline. The oracle and random-policy figures below involve no learned model: with fixed seeds they follow from the true $g$ alone, so they are identical on any machine (the GPU run reproduces them exactly). Experiment C protocol: $H = 1$, medium noise, 1,500 test contexts, danger threshold = 90th percentile of training-period log-vol.

| DGP | Drift < 0 (states) | Vol clipped (states) | Oracle: chosen $m' < 0$ | Oracle: above-median next vol | Oracle % dangerous | Random % dangerous |
|---|---|---|---|---|---|---|
| Original (v3) | 100.0% | 40.0% (floor) | 99.5% | 86.5% | 42.0% | 13.2% |
| Centered (v3.1) | 33.7% | 25.8% (6.9% floor, 18.9% ceiling) | 25.1% | 28.7% | 5.5% | 9.0% |

On the centered DGP the oracle enters dangerous states *less* often than a random policy. Risk is not the price of return in this DGP once the drift has both signs.

**Results on the centered DGP (seed 42; seeds 43–44 pending).** Experiments C–G are synthetic and use $H \le 5$, the high-SNR regime where JEPA also wins the forecasting sweep. They say nothing yet about $H = 22$, where the empirical equities live.

- **One story instead of two.** In v3, JEPA won the synthetic forecasting sweep but "lost the decision task." On the corrected DGP, in the high-SNR regime, the JEPA encoder is better at both. This does not contradict the empirical result: daily equity volatility (within-asset $R^2 \approx 0.14$) resembles the synthetic $H = 22$ regime, where JEPA already loses on forecasting. The prediction to test next is that the decision advantage also disappears when Experiments D and F are run at $H = 22$.
- **Learn the representation with JEPA, but don't plan through the learned predictor.** v3's comparison confounded representation with readout: JEPA's belief went through its non-linear predictor $P(z, a)$, while untrained features were read out linearly in the action. With the new control (the same trained Action-JEPA encoder read out linearly in $[z, a]$), the trained encoder is the best one-step planner (regret $0.0091$, 56.5% exactly optimal) ahead of untrained features ($0.0110$, 47.1%). Routing through the predictor is the worst ($0.0115$, 39.3%). All three capture over 90% of the oracle's gain over a random action, and these one-step differences are small, so they need the other two seeds. The v3 claim that "JEPA hits the optimum more often but its misses cost 3× more" does not survive the corrected DGP and the control. The likely mechanism is extrapolation: the predictor is non-linear in the action, and 75% of training actions are exactly zero, so most of the 27 grid actions are combinations it rarely saw.
- **The advantage grows with the decision horizon.** On the multi-step MDP (Experiment F), JEPA Cluster has 37% less Bellman regret than the untrained encoder ($0.278$ vs. $0.444$) and recovers $V^*$ far better (within-asset $R^2$ $0.49$ vs. $0.32$). The random linear projection's regret ($0.892$) is barely below a random policy's ($0.915$).
- **Allocation follows the same ordering.** In the portfolio QP (Experiment D), JEPA captures 56% of the oracle's expected-utility gain over equal weight, versus 44% for the untrained encoder and 31% for the random projection. In the LP scaling test (Experiment E), JEPA has the lowest regret at every portfolio size, about 20–30% below the next best, and regret does not grow with $n$.
- **Under a dynamics shift, state knowledge transfers but the decision rule does not.** JEPA's state recovery transfers best (state $R^2$ $0.223$ vs. $0.137$ for the untrained encoder), but its transferred policy is no better than the untrained one (regret $0.228$ vs. $0.211$). Retraining JEPA on the shifted world gives the best policy ($0.184$).
- **Belief-based safety filtering is not reliable.** v3's "halves dangerous entries" result (42% → 22%) was mostly the negative-drift artifact. On the centered DGP the filter cut dangerous entries for the predictor readout (9.0% → 5.9%), did nothing for untrained features (9.5% → 9.5%), and *doubled* them for the linear-readout encoder (3.6% → 7.5%). The spread of a 5-member ensemble is not a calibrated danger estimate.

<details>
<summary>v3 Experiment C output (original DGP, single seed; superseded)</summary>

Utility = conditional Sharpe $E[r']/\text{vol}'$; danger = next log-vol above the 90th percentile of all panel levels ($\tau = -2.105$); regret vs. the oracle built from the true $g$. Re-runs on GPU varied slightly (an earlier run gave Action-JEPA risk-neutral 70.7% / $0.217$ / 42.0%); v3.1 enables deterministic algorithms.

| Policy | Mean True Utility | Mean Regret | % Optimal | % Dangerous |
|---|---|---|---|---|
| Oracle (true $g$) | −0.645 | 0.000 | 100.0 | 42.8 |
| Action-JEPA Ens. risk-neutral | −0.856 | 0.210 | 70.5 | 41.7 |
| Action-JEPA Ens. safety-first | −0.902 | 0.256 | 33.7 | 21.8 |
| Untrained Ens. risk-neutral | −0.718 | 0.073 | 62.5 | 42.2 |
| Untrained Ens. safety-first | −0.770 | 0.125 | 27.8 | 23.5 |
| Random action | −1.290 | 0.644 | 3.0 | 13.9 |
| Zero action (do nothing) | −1.459 | 0.814 | 0.0 | 6.6 |

</details>

## Summary of Results

### 1. Empirical out-of-sample accuracy (45 U.S. equities and ETFs, 2022–2025, 3 seeds)

| Model Variant | Pooled $R^2$ | Within-Asset $R^2$ | IC (Pearson) | Log-MSE | Effective Rank |
|---|---|---|---|---|---|
| A. Untrained Encoder + TabICL | **0.501 ± 0.008** | **0.140 ± 0.013** | **0.709** | **0.203 ± 0.003** | 20.64 ± 0.80 |
| HAR-RV log (per-asset OLS) | 0.485 | 0.131 | 0.707 | 0.209 | — |
| Raw Features + Ridge | 0.478 | 0.110 | 0.694 | 0.212 | — |
| D. JEPA Mag + Cluster + TabICL | 0.474 ± 0.013 | 0.085 ± 0.023 | 0.692 | 0.214 ± 0.005 | 18.43 ± 0.36 |
| HAR-RV log (pooled OLS) | 0.471 | 0.096 | 0.689 | 0.215 | — |
| Raw Features + TabICL | 0.470 ± 0.004 | 0.087 ± 0.006 | 0.690 | 0.215 ± 0.001 | — |
| A. Untrained Encoder + Ridge | 0.451 ± 0.009 | 0.067 ± 0.013 | 0.674 | 0.223 ± 0.004 | 20.64 ± 0.80 |
| C. JEPA Mag + Uniform + TabICL | 0.445 ± 0.015 | 0.038 ± 0.023 | 0.674 | 0.225 ± 0.006 | 13.85 ± 0.13 |
| B. JEPA LayerNorm + Uniform + TabICL | 0.444 ± 0.012 | 0.036 ± 0.019 | 0.674 | 0.226 ± 0.005 | 13.89 ± 0.45 |
| Naive ($\log \text{RV}_{22}$) | 0.403 | 0.020 | 0.684 | 0.243 | — |

### 2. Controlled state-transition and timescale sweep ($H \in \{1, 5, 22, 60\}$ days, 3 seeds, mean ± std)

Protocol: seeds $\{42, 43, 44\}$ vary encoder init, JEPA training, and probe fits; the panel (seed 2026) is fixed; each seed's DM test uses that seed's untrained encoder. All variants are read out by TabICL. The earlier single-seed run (seed 42) reported $t = +2.25$ for Cluster at $H = 60$; across seeds this weakens to $+1.74 \pm 0.96$ (n.s.).

| $H$ | Variant | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | Latent State $R^2$ | DM $t$ vs. Untrained |
|---|---|---|---|---|---|---|
| 1 | A. Untrained Encoder | 19.63 ± 0.98 | 0.156 ± 0.002 | 0.143 ± 0.002 | 0.495 ± 0.008 | baseline |
| 1 | C. JEPA Mag + Uniform | 9.53 ± 0.49 | 0.177 ± 0.004 | 0.164 ± 0.003 | 0.517 ± 0.006 | −4.82 ± 0.81 |
| 1 | D. JEPA Mag + Cluster | 11.29 ± 1.40 | 0.172 ± 0.003 | 0.160 ± 0.003 | 0.509 ± 0.008 | −4.08 ± 0.32 |
| 1 | E. Action-JEPA + Cluster | 11.98 ± 0.99 | 0.173 ± 0.002 | 0.160 ± 0.002 | 0.498 ± 0.014 | −4.15 ± 0.78 |
| 5 | A. Untrained Encoder | 19.62 ± 0.97 | 0.378 ± 0.005 | 0.342 ± 0.004 | 0.298 ± 0.007 | baseline |
| 5 | C. JEPA Mag + Uniform | 11.52 ± 0.38 | 0.411 ± 0.003 | 0.374 ± 0.003 | 0.324 ± 0.001 | −4.85 ± 1.20 |
| 5 | D. JEPA Mag + Cluster | 11.49 ± 0.08 | 0.411 ± 0.003 | 0.375 ± 0.002 | 0.324 ± 0.003 | −5.46 ± 0.81 |
| 5 | E. Action-JEPA + Cluster | 12.22 ± 0.36 | 0.413 ± 0.004 | 0.378 ± 0.004 | 0.323 ± 0.002 | −5.87 ± 1.08 |
| 22 | A. Untrained Encoder | 19.59 ± 0.97 | 0.145 ± 0.007 | 0.087 ± 0.009 | 0.006 ± 0.005 | baseline |
| 22 | C. JEPA Mag + Uniform | 13.28 ± 0.27 | −0.130 ± 0.007 | −0.229 ± 0.011 | −0.185 ± 0.011 | +6.32 ± 0.98 |
| 22 | D. JEPA Mag + Cluster | 11.51 ± 0.38 | 0.146 ± 0.013 | 0.077 ± 0.012 | 0.003 ± 0.005 | −0.14 ± 1.14 |
| 22 | E. Action-JEPA + Cluster | 11.86 ± 0.08 | 0.151 ± 0.003 | 0.080 ± 0.001 | 0.009 ± 0.004 | −0.45 ± 0.42 |
| 60 | A. Untrained Encoder | 19.57 ± 0.95 | 0.031 ± 0.005 | −0.050 ± 0.010 | −0.019 ± 0.008 | baseline |
| 60 | C. JEPA Mag + Uniform | 13.81 ± 1.19 | −0.236 ± 0.039 | −0.445 ± 0.041 | −0.154 ± 0.025 | +5.82 ± 2.82 |
| 60 | D. JEPA Mag + Cluster | 12.93 ± 0.20 | −0.005 ± 0.019 | −0.125 ± 0.026 | −0.035 ± 0.007 | +1.74 ± 0.96 |
| 60 | E. Action-JEPA + Cluster | 13.79 ± 0.05 | 0.003 ± 0.026 | −0.121 ± 0.030 | −0.046 ± 0.007 | +0.85 ± 0.90 |

### 3. Ablation A: random linear projection vs. untrained Conv+Transformer vs. JEPA Cluster

Original DGP, medium process noise, $H = 5$, seed 42. Probes see $[z, a_t]$; TabICL is `tabicl-regressor-v2-20260212`.

| Extractor | Probe | Effective Rank | RV Pooled $R^2$ | RV Within $R^2$ | State Within $R^2$ |
|---|---|---|---|---|---|
| Random Linear | Ridge | 29.34 | 0.237 | 0.195 | 0.177 |
| Random Linear | TabICL | 29.34 | 0.287 | 0.245 | 0.236 |
| Untrained Conv+Transformer | Ridge | 18.99 | 0.339 | 0.298 | 0.253 |
| Untrained Conv+Transformer | TabICL | 18.99 | 0.373 | 0.337 | 0.290 |
| JEPA Cluster | Ridge | 11.44 | 0.389 | 0.352 | 0.318 |
| JEPA Cluster | TabICL | 11.44 | **0.414** | **0.377** | **0.329** |

### 4. Ablation B: 1st / 2nd / 3rd-order latent state recovery (Ridge probe, within-asset $R^2$)

Original DGP, medium process noise, seed 42. Order 2 = all six quadratic terms of $(l, m, p)$; order 3 = $l^3, m^3, p^3, l^2 m, l^2 p, m^2 p, l m p$.

| $H$ | Extractor | Effective Rank | Order 1 | Order 2 | Order 3 |
|---|---|---|---|---|---|
| 1 | Random Linear | 29.35 | 0.306 | 0.178 | 0.119 |
| 1 | Untrained Conv+Transformer | 19.00 | 0.413 | 0.251 | 0.164 |
| 1 | JEPA Cluster | 12.31 | **0.476** | **0.310** | **0.217** |
| 5 | Random Linear | 29.34 | 0.177 | 0.104 | 0.062 |
| 5 | Untrained Conv+Transformer | 18.99 | 0.253 | 0.151 | 0.087 |
| 5 | JEPA Cluster | 11.44 | **0.318** | **0.209** | **0.139** |
| 10 | Random Linear | 29.33 | 0.095 | 0.055 | 0.028 |
| 10 | Untrained Conv+Transformer | 18.98 | 0.132 | 0.074 | 0.033 |
| 10 | JEPA Cluster | 11.12 | **0.161** | **0.103** | **0.053** |
| 22 | Random Linear | 29.32 | 0.007 | −0.001 | −0.013 |
| 22 | Untrained Conv+Transformer | 18.95 | 0.002 | −0.014 | −0.033 |
| 22 | JEPA Cluster | 11.57 | 0.006 | −0.005 | −0.024 |

Low and high process noise give the same pattern with small shifts (JEPA Cluster order-1 at $H = 1$: 0.472 low, 0.476 medium, 0.445 high). At $H = 22$ all three extractors are at about zero. The full $3 \times 4 \times 3$ grid is printed by the script.

### 5. Decision experiments C–G (centered DGP, medium noise, seed 42)

Single seed; seeds 43–44 pending. Extractors as in Ablation A; probes are Ridge on $[z, a_t]$ unless noted.

**C. Belief-based action selection** ($H = 1$, 27-action grid, 1,500 test contexts, 5-member ensembles, safety filter $P(\text{dangerous}) \le 0.2$; utility = conditional Sharpe $E[r']/\text{vol}'$; danger threshold $\tau = 2.063$)

| Policy | Mean True Utility | Mean Regret | % Optimal | % Dangerous |
|---|---|---|---|---|
| Oracle (true $g$) | −0.0560 | 0.0000 | 100.0 | 5.5 |
| Action-JEPA Ens. risk-neutral | −0.0675 | 0.0115 | 39.3 | 9.0 |
| Action-JEPA Ens. safety-first | −0.0671 | 0.0111 | 37.3 | 5.9 |
| Action-JEPA encoder + linear-in-action, risk-neutral | **−0.0651** | **0.0091** | **56.5** | **3.6** |
| Action-JEPA encoder + linear-in-action, safety-first | −0.0662 | 0.0102 | 49.1 | 7.5 |
| Untrained Ens. risk-neutral | −0.0670 | 0.0110 | 47.1 | 9.5 |
| Untrained Ens. safety-first | −0.0671 | 0.0111 | 44.1 | 9.5 |
| Random action | −0.1779 | 0.1219 | 2.9 | 9.0 |
| Zero action (do nothing) | −0.2093 | 0.1533 | 0.0 | 10.1 |

**D. Portfolio allocation** ($H = 5$, long-only mean-variance QP, 25% cap, rebalanced every 5 days). The objective is expected utility $w'\mu - \tfrac12 w'\,\text{diag}(\text{Var})\,w$ under the true moments. "Share of gain" is each strategy's utility gain over equal weight as a share of the oracle's. The realized columns cover about 79 rebalances, so the standard error on annualized Sharpe is about 0.8; differences there are noise.

| Portfolio | Expected Utility (true) | Regret vs. Oracle | Share of Gain over 1/N | Realized Total | Realized Sharpe |
|---|---|---|---|---|---|
| Oracle QP (true moments) | 0.003364 | 0.000000 | 100% | 0.131 | 0.25 |
| JEPA Cluster QP | **0.002218** | **0.001146** | **56%** | 1.064 | 1.56 |
| Untrained Conv+Transformer QP | 0.001913 | 0.001452 | 44% | 1.087 | 1.59 |
| Random Linear QP | 0.001589 | 0.001775 | 31% | 1.102 | 1.31 |
| Equal-Weight (1/N) | 0.000788 | 0.002576 | 0% | 0.494 | 1.14 |
| Inverse-Vol (trailing 22d) | −0.005330 | 0.008694 | — | −0.426 | −3.18 |

**E. LP decision-complexity scaling** ($\max_w w'\hat\mu$ s.t. $\sum w = 1$, $0 \le w \le 2/n$; 8 random subsets per size, 1 at $n = 20$). Mean regret vs. the LP on true expected returns, ×10⁻⁴; top-1 hit rate in parentheses.

| $n$ | Random Linear | Untrained Conv+Transformer | JEPA Cluster | Equal-Weight |
|---|---|---|---|---|
| 2 | 1.79 (87.0%) | 1.78 (88.6%) | **1.25 (90.8%)** | 29.0 |
| 4 | 1.77 (70.1%) | 1.67 (71.0%) | **1.27 (75.0%)** | 33.7 |
| 8 | 1.56 (60.9%) | 1.65 (62.5%) | **1.28 (64.4%)** | 36.2 |
| 16 | 1.97 (35.0%) | 1.86 (30.2%) | **1.43 (39.7%)** | 36.2 |
| 20 | 1.97 (29.1%) | 1.77 (27.8%) | **1.28 (35.4%)** | 36.7 |

**F. RL emulation** (per-asset value iteration, $\gamma = 0.9$, reward = conditional Sharpe of the next state; Monte Carlo Bellman regret, 64 samples; $\pi^*$ labels are about 94–97% reproducible across Monte Carlo seeds)

| Policy / Extractor | MC Bellman Regret | $\pi^*$ Accuracy | $V^*$ Within $R^2$ | $V^*$ Pooled $R^2$ |
|---|---|---|---|---|
| Oracle (value iteration) | 0.000 | 100.0% | — | — |
| JEPA Cluster | **0.278** | **35.1%** | **0.491** | **0.428** |
| Untrained Conv+Transformer | 0.444 | 31.5% | 0.317 | 0.335 |
| Random Linear | 0.892 | 20.5% | 0.115 | 0.157 |
| Random action | 0.915 | 3.6% | — | — |
| Zero action (do nothing) | 0.943 | 0.1% | — | — |

**G. Dynamics shift** ($g_{\text{shift}} = g + 0.1\tanh(s - \bar s)$ on $(l, m)$, centered on the original mean $(-0.803, 0.635)$; regret under the shifted MDP's oracle). "Transferred" = A-trained extractor and probes applied to B with A's normalization; "refit" = same frozen extractor, probes refit on B; "retrained" = JEPA trained on B.

| Extractor | Transferred: Regret | Transferred: $\pi^*_B$ Acc. | Transferred: State $R^2$ | On B: Regret | On B: $\pi^*_B$ Acc. | On B: State $R^2$ |
|---|---|---|---|---|---|---|
| Random Linear (refit) | 0.412 | 29.0% | 0.057 | 0.211 | 38.1% | 0.090 |
| Untrained Conv+Transformer (refit) | **0.211** | **34.0%** | 0.137 | 0.199 | 40.4% | 0.225 |
| JEPA Cluster (retrained) | 0.228 | 31.8% | **0.223** | **0.184** | **43.3%** | **0.299** |
| Random action | 1.492 | 3.7% | — | — | — | — |
| Zero action (do nothing) | 1.722 | 0.0% | — | — | — | — |

#### Changes from v3 that affect these numbers

| Experiment | Change in v3.1 |
|---|---|
| All (C–G) | Centered drift DGP (see above); `--decision-dgp original` keeps the v3 DGP |
| C. Belief-based action selection | Danger threshold from training-period states only; new control row (trained Action-JEPA encoder, linear-in-action readout) |
| D. Portfolio allocation | v3's "oracle" was a heuristic rule, so learned strategies showed negative regret. Every strategy now solves the same exact mean-variance QP, and the oracle plugs in the true moments (regret ≥ 0). Fixed the true H-day variance, which v3 multiplied by $H$ twice |
| E. LP scaling | No method change; $n = 20$ is a single subset (the full panel) |
| F. RL emulation | Value iteration per asset (each asset has its own $\beta$ in $g$), vectorized, with common random numbers; "exact Bellman regret" renamed Monte Carlo Bellman regret (64 samples). The $\pi^*$ labels agree on 94–97% of states across Monte Carlo seeds, which caps attainable $\pi^*$ accuracy |
| G. Dynamics shift | Transferred models see B's features normalized with A's training statistics; Random Linear and Untrained keep the same frozen extractor and only refit probes on B (v3 re-drew the untrained encoder) |

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── paper/
│   ├── Arruda_RandomFeatures_JEPA_TabICL_Volatility_v3.md
│   └── Arruda_RandomFeatures_JEPA_TabICL_Volatility_v3.docx
└── scripts/
    ├── study3_state_timescale_experiments.py   # Experiments 1–2: state-transition recovery + H ∈ {1,5,22,60} timescale sweep (3 seeds)
    └── study3_ablations.py                     # Ablations A/B; Experiments C (belief-based action selection),
                                                # D (portfolio allocation), E (LP decision-complexity scaling),
                                                # F (RL emulation), G (dynamics shift). Self-contained, Colab-runnable
```

## Quickstart

```bash
git clone https://github.com/DomArruda/Volatility-Forecasting-with-Foundation-Models.git
cd Volatility-Forecasting-with-Foundation-Models
pip install -r requirements.txt
python scripts/study3_state_timescale_experiments.py

# Ablations + decision experiments; one run per seed (each table is single-seed)
python scripts/study3_ablations.py --quick            # smoke test, a few minutes
for s in 42 43 44; do python scripts/study3_ablations.py --seed $s | tee ablations_seed$s.txt; done
```

`study3_ablations.py` flags: `--seed` (default 42), `--decision-dgp {centered,original}` (default `centered`), `--quick`. Install `tabicl` before reporting TabICL rows; without it the script uses an RBF fallback and labels those rows `rbf-fb`.

**Google Colab:** upload `scripts/study3_ablations.py` to the Colab VM and run `!python study3_ablations.py --seed 42`. Alternatively, open it as a notebook (cells split on `# %%`) and Run all; flags can't be passed that way, so edit the defaults in `_parse_args`.

## Limitations

- Ablations A/B and the decision experiments are single-seed per run; aggregate the 42/43/44 runs before drawing significance conclusions. The one-step gaps in Experiment C (regret 0.009–0.012) are small enough that a single seed cannot rank the three readouts reliably.
- The decision experiments run only at $H \le 5$, the high-SNR synthetic regime. Whether the JEPA decision advantage holds at $H = 22$, the regime that resembles empirical equities, is untested.
- Experiment D's realized-return columns span about 79 rebalances (Sharpe standard error about 0.8); only the expected-utility comparison is informative.
- Both synthetic DGPs have a wide log-vol distribution, so some states hit the volatility clip (original: 40% at the floor; centered: 7% at the floor and 19% at the ceiling). Clipped states carry no return information about $l$.
- Three seeds give rough standard deviations; "not significant across seeds" for $t = +1.74 \pm 0.96$ is a statement about seed-to-seed variation, not a formal multi-seed test.


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.