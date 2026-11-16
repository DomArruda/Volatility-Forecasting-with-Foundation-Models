"""
Study 3 - Targeted Ablations and Decision Experiments (extends, and does not modify,
the original Study-3 codebase).

SELF-CONTAINED GOOGLE-COLAB RUNNER: this single file embeds verbatim copies of the
existing Study-3 utilities (encoder, JEPA training loop, probes, metrics, data
generation) so it runs with NO other files present. The originals in
scripts/study3_state_timescale_experiments.py remain the reference implementation;
keep the copies below in sync with it.

ABLATION A (Table 1): does the structure of the untrained Conv+Transformer matter, or would
    any random projection of the same width work? Random linear projection (out_dim=33) vs
    untrained Conv+Transformer vs JEPA Cluster; medium noise, H=5; Ridge and TabICL probes.

ABLATION B (Table 2): how fast does recovery of 1st/2nd/3rd-order latent-state terms degrade
    with horizon H in {1,5,10,22} and process noise {low, medium, high}? Ridge probe.

    Ablations A and B use the ORIGINAL DGP, so they line up with Experiments 1-2.

DECISION EXPERIMENTS C-G use the DECISION DGP (--decision-dgp, default "centered").
    In the original DGP the drift equation couples to the raw log-vol level,
    tanh(a_regime + 0.3 l); with l around -4.7 that pins the drift m below zero in every
    state, so the conditional Sharpe m/vol is always negative and is maximized by RAISING
    vol. It also leaves ~40% of states below the vol clip floor (l < -4.5), where returns
    carry no information about l. The centered DGP couples to the asset's own baseline,
    tanh(a_regime + 0.3 (l - beta)); everything else in g is unchanged. Each run prints
    both panels' diagnostics. Measured at the default panel (medium noise): drift < 0 in
    100% -> 34% of states; vol clipped in 40% (all at the floor) -> 26% (7% floor, 19%
    ceiling, l > 1). The log-vol spread itself is wide in both DGPs; bounding it would
    need a further change to g and is left as a modeling choice. "--decision-dgp original" runs C-G on the v3 DGP (numbers
    differ slightly from v3 because of the fixes below).

EXPERIMENT C (Table 3, belief-space planning): 27-point action grid; 5-member ensembles form a
    belief over the next state (mean = expected dynamics, spread = epistemic risk). Three
    readouts: Action-JEPA predictor P(z, a); the same trained Action-JEPA ENCODER read out
    linearly in the action [z, a] (control that separates "representation" from "nonlinear
    action pathway"); untrained encoder read out on [z, a]. Risk-neutral and safety-first
    (filter P(dangerous) <= tol, then max conditional Sharpe) policies, scored against the
    oracle built from the true g.

EXPERIMENT D (Table 4, portfolio allocation): long-only mean-variance allocation with a 25%
    cap, rebalanced every H=5 days. Every strategy solves the SAME QP,
        max_w  w'mu - 0.5 w'diag(var) w   s.t.  sum(w) = 1, 0 <= w <= cap,
    exactly; the oracle plugs in the true moments (computed from the realized future DGP
    states), so regret vs oracle is >= 0 by construction.

EXPERIMENT E (Table 5, LP decision-complexity scaling): max_w w'mu_hat s.t. sum(w)=1,
    0 <= w <= 2/n, for n in {2,4,8,16,20}; regret vs the LP on true expected returns.

EXPERIMENT F (Table 6, RL emulation): the MDP induced by g (27 actions, process noise,
    reward = conditional Sharpe of the next state, gamma = 0.9) is solved by value iteration
    PER ASSET (each asset has its own beta in g). Q-values use common random numbers
    shared across actions (sample-average approximation), so value iteration is a
    deterministic contraction. Extractors are probed for V* and pi*; probed policies are
    scored by Monte Carlo Bellman regret (64 shared noise samples per state; pi* labels
    agree ~94-97% across Monte Carlo seeds, which caps attainable pi* accuracy).

EXPERIMENT G (Table 7, dynamics shift): g_shift = g + bounded tanh repulsion from the original
    (l, m) mean. A-trained extractors + probes are transferred to the shifted panel, whose
    features are normalized with A's training statistics (so the shift is not hidden by
    re-standardization), and compared with probes refit on B (Random Linear, Untrained:
    same frozen extractor) and a JEPA retrained on B.

How to run in Google Colab (any ONE of these):
    1. Upload the file to the Colab VM and run:  !python study3_ablations.py --seed 42
    2. File -> Upload notebook -> this .py file (cells split on '# %%') -> Run all.
       Flags cannot be passed this way; edit the defaults in _parse_args instead.
    3. Once pushed to GitHub: colab.research.google.com -> GitHub tab -> this file.

Flags:
    --seed S             encoder init / JEPA training / probe seed (default 42). Every table
                         is single-seed; run 42, 43, 44 and aggregate for mean +- std.
    --decision-dgp D     "centered" (default) or "original" (the v3 DGP) for Experiments C-G.
    --quick              tiny smoke test (a few minutes on CPU; numbers are meaningless).

Runtime: 12 JEPA trainings for Ablation B, 5 Action-JEPA trainings for C, one extra JEPA for
the centered decision panel and one for the shifted world (G): roughly 60-100 minutes on a
Colab CPU, much less on GPU. Value iteration is vectorized and takes seconds.
`tabicl` is optional; without it the "TabICL" probe is an RBF random-features + Ridge
fallback, and the tables label those rows "rbf-fb" so they are never reported as TabICL.
"""
%pip install tabicl
# %%
# ============================================================================
# SETUP & IMPORTS (verbatim from study3_state_timescale_experiments.py)
# ============================================================================

import itertools
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from scipy.special import erfc
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# Attempt to import TabICL; provide a faithful In-Context Kernel/Ensemble fallback if absent
try:
    from tabicl import TabICLRegressor
    HAS_TABICL = True
except ImportError:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline
    HAS_TABICL = False


# %%
# ============================================================================
# EXISTING STUDY-3 UTILITIES - VERBATIM COPY (keep in sync with
# scripts/study3_state_timescale_experiments.py). Nothing here is changed except
# optional kwargs whose defaults reproduce the original behavior bit-for-bit:
# state_noise / transition_fn on generate_panel_with_interventions, and
# transition_fn / norm_stats on build_horizon_dataset (which also returns its
# normalization stats under "norm_stats").
# ============================================================================

# ----------------------------------------------------------------------------
# 1. UTILITIES: SEEDS, EFFECTIVE RANK (Roy & Vetterli, 2007)
# ----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_effective_rank(embeddings: np.ndarray) -> float:
    """
    Roy & Vetterli (2007) effective rank of out-of-sample embeddings.
    erank(Z) = exp(- sum_i p_i log(p_i)), where p_i = sigma_i / sum_j sigma_j.
    """
    Z = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Z, full_matrices=False)
    s = s[s > 1e-12]
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ----------------------------------------------------------------------------
# 2. SYNTHETIC STATE-FUNCTION & MULTI-TIMESCALE ENVIRONMENT
# ----------------------------------------------------------------------------

@dataclass
class SimConfig:
    n_assets: int = 20
    n_days: int = 1500
    context_len: int = 40
    n_features: int = 8
    n_actions: int = 3
    n_states: int = 3


def transition_function(state_t: np.ndarray, action_t: np.ndarray, asset_beta: float) -> np.ndarray:
    """
    Deterministic part of the daily state-transition function g(s_t, a_t).
    state_t: [l_t (log-vol level), m_t (drift/regime), p_t (persistence)]
    action_t: [a_shock (vol impulse), a_regime (regime shift), a_damp (mean-reversion control)]
    """
    l_t, m_t, p_t = state_t[0], state_t[1], state_t[2]
    a_shock, a_regime, a_damp = action_t[0], action_t[1], action_t[2]

    eff_persistence = 0.88 + 0.10 * np.tanh(p_t - 1.2 * a_damp)

    nonlinear_kick = (
        0.45 * a_shock * (1.0 + 0.5 * np.tanh(l_t))
        + 0.30 * np.sin(1.5 * a_regime + 0.5 * m_t)
        - 0.25 * a_damp * l_t
    )

    l_next = (1.0 - eff_persistence) * asset_beta + eff_persistence * l_t + nonlinear_kick
    m_next = 0.85 * m_t + 0.60 * np.tanh(a_regime + 0.3 * l_t) - 0.20 * a_shock
    p_next = 0.90 * p_t - 0.50 * a_damp + 0.15 * np.abs(a_shock)

    return np.array([l_next, m_next, p_next], dtype=np.float32)


def generate_panel_with_interventions(
    cfg: SimConfig,
    seed: int = 0,
    state_noise: Tuple[float, float, float] = (0.06, 0.05, 0.03),
    transition_fn=None,
) -> Dict[str, np.ndarray]:
    """
    Simulates a panel of assets with latent multi-scale states and daily functional
    interventions u_t, plus 8 observable Study-3-style features.
    state_noise: per-coordinate std of the Gaussian process noise on the latent
    state transition (l, m, p). Default is the original medium level; the ablation
    sweep varies it without changing the default stream.
    """
    rng = np.random.default_rng(seed)
    N, T = cfg.n_assets, cfg.n_days

    asset_betas = rng.uniform(-2.2, -1.0, size=N).astype(np.float32)

    states = np.zeros((N, T, cfg.n_states), dtype=np.float32)
    actions = np.zeros((N, T, cfg.n_actions), dtype=np.float32)
    returns = np.zeros((N, T), dtype=np.float32)
    features = np.zeros((N, T, cfg.n_features), dtype=np.float32)

    for i in range(N):
        s = np.array([asset_betas[i], 0.0, 0.0], dtype=np.float32)
        for t in range(T):
            event_mask = rng.binomial(1, 0.25)
            a_t = event_mask * rng.normal(0.0, 1.0, size=cfg.n_actions).astype(np.float32)
            actions[i, t] = a_t
            states[i, t] = s

            vol_t = np.exp(np.clip(s[0], -4.5, 1.0)) / np.sqrt(252.0)
            r_t = 0.0005 * s[1] + vol_t * rng.standard_normal()
            returns[i, t] = r_t

            s_det = (transition_fn if transition_fn is not None else transition_function)(s, a_t, asset_betas[i])
            noise = rng.normal([0.0, 0.0, 0.0], list(state_noise)).astype(np.float32)
            s = s_det + noise

        r = returns[i]
        r2 = r ** 2
        eps = 1e-8
        for t in range(60, T):
            ret_1d = r[t]
            ret_5d = np.sum(r[t - 4:t + 1])
            ret_20d = np.sum(r[t - 19:t + 1])
            log_abs_r = np.log(np.sqrt(252.0) * np.abs(ret_1d) + eps)
            rv_5 = np.log(np.sqrt(252.0 * np.mean(r2[t - 4:t + 1])) + eps)
            rv_22 = np.log(np.sqrt(252.0 * np.mean(r2[t - 21:t + 1])) + eps)
            rv_ratio = rv_5 - rv_22
            z_60 = (ret_1d - np.mean(r[t - 59:t + 1])) / (np.std(r[t - 59:t + 1]) + eps)
            features[i, t] = [ret_1d, ret_5d, ret_20d, log_abs_r, rv_5, rv_22, rv_ratio, z_60]

    return {
        "features": features,
        "actions": actions,
        "states": states,
        "returns": returns,
        "asset_betas": asset_betas,
    }


def build_horizon_dataset(
    panel: Dict[str, np.ndarray],
    context_len: int = 40,
    horizon: int = 5,
    train_end_day: int = 1000,
    transition_fn=None,
    norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Constructs context windows X_{t-L+1:t}, future target windows X_{t+1:t+H},
    actions applied on decision day t, future state s_{t+H},
    and log H-day realized volatility target y_{t+H}, with an H-day embargo split.
    transition_fn: the g that generated the panel (for the s_det1 target); default = transition_function.
    norm_stats: (mean, std) to normalize features with; default = this panel's training slice.
    Both defaults reproduce the original behavior exactly.
    """
    feats = panel["features"]
    acts = panel["actions"]
    states = panel["states"]
    rets = panel["returns"]
    betas = panel["asset_betas"]
    N, T, F_dim = feats.shape

    start_t = 60 + context_len
    max_t = T - horizon - 1

    if norm_stats is None:
        train_slice = feats[:, start_t:train_end_day, :].reshape(-1, F_dim)
        f_mean = train_slice.mean(axis=0, keepdims=True)
        f_std = train_slice.std(axis=0, keepdims=True) + 1e-8
    else:
        f_mean, f_std = norm_stats
    g = transition_fn if transition_fn is not None else transition_function
    feats_norm = np.clip((feats - f_mean) / f_std, -8.0, 8.0)

    splits = {"train": [], "test": []}
    for i in range(N):
        for t in range(start_t, max_t):
            if t + horizon < train_end_day:
                split = "train"
            elif t >= train_end_day:
                split = "test"
            else:
                continue

            ctx_win = feats_norm[i, t - context_len + 1:t + 1]
            fut_raw = feats_norm[i, t + 1:t + 1 + horizon]

            act_t = acts[i, t]
            s_t = states[i, t]
            s_next_det = g(s_t, act_t, betas[i])
            s_horizon = states[i, t + horizon]

            r2_fut = rets[i, t + 1:t + 1 + horizon] ** 2
            y_rv = float(np.log(np.sqrt(252.0 * np.mean(r2_fut)) + 1e-8))

            splits[split].append((ctx_win, fut_raw, act_t, s_t, s_next_det, s_horizon, y_rv, i, t))

    out = {}
    for sp in ["train", "test"]:
        rows = splits[sp]
        out[f"X_ctx_{sp}"] = np.stack([r[0] for r in rows]).astype(np.float32)
        out[f"X_fut_{sp}"] = np.stack([r[1] for r in rows]).astype(np.float32)
        out[f"act_{sp}"] = np.stack([r[2] for r in rows]).astype(np.float32)
        out[f"s_curr_{sp}"] = np.stack([r[3] for r in rows]).astype(np.float32)
        out[f"s_det1_{sp}"] = np.stack([r[4] for r in rows]).astype(np.float32)
        out[f"s_hor_{sp}"] = np.stack([r[5] for r in rows]).astype(np.float32)
        out[f"y_rv_{sp}"] = np.array([r[6] for r in rows], dtype=np.float32)
        out[f"asset_{sp}"] = np.array([r[7] for r in rows], dtype=np.int64)
        out[f"date_{sp}"] = np.array([r[8] for r in rows], dtype=np.int64)
    out["norm_stats"] = (f_mean, f_std)
    return out


# ----------------------------------------------------------------------------
# 3. JEPA ENCODER, ACTION-CONDITIONED PREDICTOR & CLUSTER BATCH SAMPLER
# ----------------------------------------------------------------------------

class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_features: int = 8, d_model: int = 32, n_heads: int = 4, n_layers: int = 3, mag_preserving: bool = True):
        super().__init__()
        self.mag_preserving = mag_preserving
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out_dim = d_model + (1 if mag_preserving else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.transformer(h)
        pooled = 0.5 * (h.mean(dim=1) + h[:, -1, :])
        normed = self.ln(pooled)
        if self.mag_preserving:
            mag = torch.log1p(torch.norm(pooled, p=2, dim=-1, keepdim=True))
            return torch.cat([normed, mag], dim=-1)
        return normed


class LatentPredictor(nn.Module):
    def __init__(self, embed_dim: int = 33, action_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        in_dim = embed_dim + action_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, z: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.action_dim > 0 and a is not None:
            x = torch.cat([z, a], dim=-1)
        else:
            x = z
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(h)


class ClusterBatchSampler(Sampler):
    def __init__(self, n_samples: int, batch_size: int = 256, n_clusters: int = 16, cluster_frac: float = 0.85):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.cluster_frac = cluster_frac
        self.labels = np.zeros(n_samples, dtype=np.int64)

    def update_clusters(self, embeddings: np.ndarray, seed: int = 0):
        km = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=seed)
        self.labels = km.fit_predict(embeddings)

    def __iter__(self):
        n_c = int(round(self.batch_size * self.cluster_frac))
        n_r = self.batch_size - n_c

        batches = []
        leftovers = []
        cluster_ids = np.unique(self.labels)
        np.random.shuffle(cluster_ids)

        for cid in cluster_ids:
            c_idx = np.where(self.labels == cid)[0].copy()
            np.random.shuffle(c_idx)
            outside_idx = np.where(self.labels != cid)[0]
            ptr = 0
            while ptr + n_c <= len(c_idx):
                core = c_idx[ptr:ptr + n_c]
                ptr += n_c
                rand_fill = np.random.choice(outside_idx, size=n_r, replace=False)
                batches.append(np.concatenate([core, rand_fill]))
            if ptr < len(c_idx):
                leftovers.append(c_idx[ptr:])

        if leftovers:
            pool = np.concatenate(leftovers)
            np.random.shuffle(pool)
            for i in range(0, len(pool) - self.batch_size + 1, self.batch_size):
                batches.append(pool[i:i + self.batch_size])

        np.random.shuffle(batches)
        for b in batches:
            yield b.tolist()

    def __len__(self):
        return self.n_samples // self.batch_size


def jepa_vicreg_loss(pred: torch.Tensor, target: torch.Tensor, mag_preserving: bool = True) -> torch.Tensor:
    mse_loss = F.mse_loss(pred, target)
    z_reg = pred[:, :-1] if mag_preserving else pred
    B, D = z_reg.shape
    if B < 4:
        return 12.0 * mse_loss

    std = torch.sqrt(z_reg.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))

    z_centered = z_reg - z_reg.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (B - 1)
    off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    cov_loss = torch.mean(off_diag ** 2)

    return 12.0 * mse_loss + 10.0 * var_loss + 1.0 * cov_loss


def train_jepa_model(
    X_ctx_train: np.ndarray,
    X_fut_train: np.ndarray,
    act_train: np.ndarray,
    use_action_predictor: bool = False,
    use_cluster_sampler: bool = False,
    mag_preserving: bool = True,
    epochs: int = 12,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[TimeSeriesEncoder, LatentPredictor]:
    set_seed(seed)
    ctx_enc = TimeSeriesEncoder(mag_preserving=mag_preserving).to(device)
    tgt_enc = TimeSeriesEncoder(mag_preserving=mag_preserving).to(device)
    tgt_enc.load_state_dict(ctx_enc.state_dict())
    tgt_enc.eval()
    for p in tgt_enc.parameters():
        p.requires_grad = False

    act_dim = act_train.shape[1] if use_action_predictor else 0
    predictor = LatentPredictor(embed_dim=ctx_enc.out_dim, action_dim=act_dim).to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_ctx_train),
        torch.from_numpy(X_fut_train),
        torch.from_numpy(act_train),
    )

    sampler = ClusterBatchSampler(len(dataset), batch_size=batch_size) if use_cluster_sampler else None
    optimizer = torch.optim.AdamW(
        list(ctx_enc.parameters()) + list(predictor.parameters()),
        lr=1.5e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema_tau = 0.996

    for epoch in range(epochs):
        if use_cluster_sampler and (epoch == 1 or (epoch > 1 and epoch % 4 == 0)):
            with torch.no_grad():
                all_fut = torch.from_numpy(X_fut_train).to(device)
                tgt_embs = []
                for idx in range(0, len(all_fut), 512):
                    tgt_embs.append(tgt_enc(all_fut[idx:idx + 512]).cpu().numpy())
                sampler.update_clusters(np.vstack(tgt_embs), seed=seed + epoch)

        if use_cluster_sampler and epoch >= 1:
            loader = DataLoader(dataset, batch_sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        ctx_enc.train()
        predictor.train()
        for bx_ctx, bx_fut, bact in loader:
            bx_ctx, bx_fut, bact = bx_ctx.to(device), bx_fut.to(device), bact.to(device)
            z_ctx = ctx_enc(bx_ctx)
            pred_z = predictor(z_ctx, bact if use_action_predictor else None)
            with torch.no_grad():
                tgt_z = tgt_enc(bx_fut)

            loss = jepa_vicreg_loss(pred_z, tgt_z, mag_preserving=mag_preserving)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p_c, p_t in zip(ctx_enc.parameters(), tgt_enc.parameters()):
                    p_t.data.mul_(ema_tau).add_(p_c.data, alpha=1.0 - ema_tau)

        scheduler.step()

    ctx_enc.eval()
    predictor.eval()
    return ctx_enc, predictor


@torch.no_grad()
def extract_representations(
    ctx_enc: TimeSeriesEncoder,
    predictor: LatentPredictor,
    X_ctx: np.ndarray,
    act: Optional[np.ndarray] = None,
    use_action: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    ctx_enc.eval()
    predictor.eval()
    outs = []
    for i in range(0, len(X_ctx), 512):
        bx = torch.from_numpy(X_ctx[i:i + 512]).to(device)
        z = ctx_enc(bx)
        if use_action and act is not None:
            ba = torch.from_numpy(act[i:i + 512]).to(device)
            p = predictor(z, ba)
        else:
            p = predictor(z, None)
        outs.append(p.cpu().numpy())
    return np.vstack(outs)


# ----------------------------------------------------------------------------
# 4. PROBES: TABICL (WITH SUPPORT STRIDE) & RIDGE + WITHIN-ASSET R2
# ----------------------------------------------------------------------------

def fit_predict_probe(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    probe_type: str = "tabicl",
    support_stride: int = 5,
    max_support: int = 3000,
    seed: int = 0,
) -> np.ndarray:
    scaler = StandardScaler()
    Z_tr_s = scaler.fit_transform(Z_train)
    Z_te_s = scaler.transform(Z_test)

    if probe_type == "ridge":
        model = Ridge(alpha=10.0, random_state=seed)
        model.fit(Z_tr_s, y_train)
        return model.predict(Z_te_s)

    idx_strided = np.arange(0, len(Z_tr_s), support_stride)
    if len(idx_strided) > max_support:
        rng = np.random.default_rng(seed)
        idx_strided = rng.choice(idx_strided, size=max_support, replace=False)

    X_sup = Z_tr_s[idx_strided]
    y_sup = y_train[idx_strided]

    if HAS_TABICL:
        if y_sup.ndim == 1:
            reg = TabICLRegressor(n_estimators=4, random_state=seed)
            reg.fit(X_sup, y_sup)
            return reg.predict(Z_te_s)
        else:
            preds = []
            for d in range(y_sup.shape[1]):
                reg = TabICLRegressor(n_estimators=4, random_state=seed)
                reg.fit(X_sup, y_sup[:, d])
                preds.append(reg.predict(Z_te_s))
            return np.column_stack(preds)
    else:
        preds = []
        for est in range(4):
            pipe = make_pipeline(
                RBFSampler(gamma=1.0 / Z_tr_s.shape[1], n_components=256, random_state=seed + est),
                Ridge(alpha=5.0)
            )
            pipe.fit(X_sup, y_sup)
            preds.append(pipe.predict(Z_te_s))
        return np.mean(preds, axis=0)


def within_asset_r2(y_true: np.ndarray, y_pred: np.ndarray, asset_ids: np.ndarray) -> float:
    yt_dem = y_true.copy()
    yp_dem = y_pred.copy()
    for aid in np.unique(asset_ids):
        mask = asset_ids == aid
        yt_dem[mask] -= np.mean(yt_dem[mask], axis=0)
        yp_dem[mask] -= np.mean(yp_dem[mask], axis=0)
    return float(r2_score(yt_dem, yp_dem))


# %%
# ============================================================================
# NEW CODE (everything above is the existing Study-3 machinery copied verbatim
# so this single file runs standalone in Colab).
# ============================================================================

import argparse
import os
from typing import Callable


def _parse_args():
    p = argparse.ArgumentParser(description="Study 3 ablations and decision experiments")
    p.add_argument("--seed", type=int, default=42,
                   help="encoder init / JEPA training / probe seed; run 42, 43, 44 for 3-seed tables")
    p.add_argument("--decision-dgp", choices=["centered", "original"], default="centered",
                   help="DGP for Experiments C-G ('original' = the v3 DGP)")
    p.add_argument("--quick", action="store_true", help="tiny smoke test; numbers are meaningless")
    args, _ = p.parse_known_args()  # tolerate Jupyter/Colab kernel arguments
    return args


ARGS = _parse_args()

# Protocol constants (identical to the original Study-3 experiments unless --quick)
PANEL_SEED = 2026          # same panel seed as the original main script
PROBE_SEED = ARGS.seed     # 42 = Experiment 2 seed (untrained encoder, JEPA, probes)
CONTEXT_LEN = 40
N_ASSETS = 20
N_DAYS, TRAIN_END_DAY = (500, 350) if ARGS.quick else (1400, 1000)
JEPA_EPOCHS = 1 if ARGS.quick else 10
N_ENSEMBLE = 2 if ARGS.quick else 5
VI_ITERS = 5 if ARGS.quick else 30
RANDOM_PROJ_OUT_DIM = 33   # matches TimeSeriesEncoder(mag_preserving=True).out_dim
PROBE_LABEL = "tabicl" if HAS_TABICL else "rbf-fb"   # the RBF fallback is NOT TabICL

# Process-noise levels on the latent state transition (l, m, p); medium = original
NOISE_LEVELS: Dict[str, Tuple[float, float, float]] = {
    "low": (0.02, 0.015, 0.01),
    "medium": (0.06, 0.05, 0.03),
    "high": (0.15, 0.12, 0.08),
}
ABLATION_HORIZONS = [1, 5, 10, 22]

# Shared decision machinery (Experiments C-G): 27-point action grid and the DGP process noise
ACTION_GRID = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=np.float32)
ZERO_ACTION = int(np.argmin(np.linalg.norm(ACTION_GRID, axis=1)))
PROCESS_NOISE = np.array(NOISE_LEVELS["medium"], dtype=np.float64)
# Registries filled as experiments run: trained extractors (for cross-panel transfer) and solved MDPs.
EXTRACTOR_MODELS: Dict[Tuple[str, int], dict] = {}
_MDP_CACHE: Dict[str, dict] = {}


def _ens_seed(base: int, member: int) -> int:
    """Ensemble-member seed; equals the v3 seeds (base + member) at the default --seed 42."""
    return base + member + 1000 * (PROBE_SEED - 42)


# ----------------------------------------------------------------------------
# DECISION DGP: vectorized g and the "world" used by Experiments C-G
# ----------------------------------------------------------------------------

def transition_function_vec(S, A, beta, centered_drift: bool = False) -> np.ndarray:
    """
    Vectorized g(s, a) over any leading dims (S, A: (..., 3); beta broadcastable).
    centered_drift=False is the original transition_function. centered_drift=True changes
    ONLY the drift equation's vol coupling, tanh(a_regime + 0.3 l) -> tanh(a_regime + 0.3 (l - beta)),
    so the drift no longer sits permanently below zero.
    """
    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    l, m, p = S[..., 0], S[..., 1], S[..., 2]
    a_shock, a_regime, a_damp = A[..., 0], A[..., 1], A[..., 2]

    eff_persistence = 0.88 + 0.10 * np.tanh(p - 1.2 * a_damp)
    nonlinear_kick = (
        0.45 * a_shock * (1.0 + 0.5 * np.tanh(l))
        + 0.30 * np.sin(1.5 * a_regime + 0.5 * m)
        - 0.25 * a_damp * l
    )
    l_next = (1.0 - eff_persistence) * beta + eff_persistence * l + nonlinear_kick
    coupling = 0.3 * ((l - beta) if centered_drift else l)
    m_next = 0.85 * m + 0.60 * np.tanh(a_regime + coupling) - 0.20 * a_shock
    p_next = 0.90 * p - 0.50 * a_damp + 0.15 * np.abs(a_shock)
    return np.stack([l_next, m_next, p_next], axis=-1)


@dataclass
class World:
    name: str
    g: Callable       # scalar g(s, a, beta) -> float32 (3,), used to simulate panels
    g_vec: Callable   # vectorized g(S, A, beta), used by oracles and value iteration


def make_world(dgp: str) -> World:
    if dgp == "original":
        return World("original", transition_function,
                     lambda S, A, b: transition_function_vec(S, A, b, centered_drift=False))

    def g_centered(s, a, b):
        return transition_function_vec(s, a, b, centered_drift=True).astype(np.float32)

    return World("centered", g_centered,
                 lambda S, A, b: transition_function_vec(S, A, b, centered_drift=True))


def _world_key(world: World) -> str:
    """Feature-cache key for the medium-noise decision panel (shared with Ablation A if original)."""
    return "medium" if world.name == "original" else f"medium-{world.name}"


def state_utility(s_next: np.ndarray) -> np.ndarray:
    """Conditional Sharpe: expected next-day return per unit next-day vol.
    Matches the panel construction: E[r_{t+1}] = 0.0005 * m', vol = exp(l')/sqrt(252)."""
    l, m = s_next[..., 0], s_next[..., 1]
    vol = np.exp(np.clip(l, -4.5, 1.0)) / np.sqrt(252.0)
    return 0.0005 * m / (vol + 1e-12)


def print_dgp_diagnostics(name: str, panel: Dict[str, np.ndarray]):
    """The two properties that made the original DGP's decision experiments misleading."""
    s = panel["states"][:, 100:].reshape(-1, 3)
    l, m = s[:, 0], s[:, 1]
    vol = np.exp(np.clip(l, -4.5, 1.0)) / np.sqrt(252.0)
    print(f"  {name:<22} mean l {l.mean():6.2f} | mean m {m.mean():6.2f} | "
          f"m < 0: {100 * np.mean(m < 0):5.1f}% | vol clipped: {100 * np.mean(l < -4.5):4.1f}% below / "
          f"{100 * np.mean(l > 1.0):4.1f}% above | "
          f"median |drift|/vol {np.median(np.abs(0.0005 * m) / vol):.3f}")


# ----------------------------------------------------------------------------
# EXTRACTORS
# ----------------------------------------------------------------------------

class RandomLinearProjector(nn.Module):
    """
    ABLATION A baseline: flattens the context window (B, T, F) -> (B, T*F) and applies
    one fixed random matrix of shape (T*F, out_dim). Weights are drawn once and frozen,
    so it has the same stateless interface as the untrained Conv+Transformer encoder.
    """

    def __init__(self, context_len: int = 40, n_features: int = 8, out_dim: int = 33, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W",
            torch.randn(context_len * n_features, out_dim, generator=gen)
            / np.sqrt(context_len * n_features),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1) @ self.W


class IdentityPredictor(nn.Module):
    """No-op predictor so extract_representations returns the raw encoder/projector output."""

    def forward(self, z: torch.Tensor, a=None) -> torch.Tensor:
        return z


def build_extractor_features(
    data: Dict[str, np.ndarray],
    panel_key: str,
    horizon: int,
    device: str = "cpu",
    cache: Dict[Tuple[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Train/test representations {extractor: (Z_train, Z_test)} for the three extractors,
    using the exact protocols of the original experiments:
      - Random Linear: frozen projector (seed = PROBE_SEED)
      - Untrained Conv+Transformer: set_seed(PROBE_SEED), untrained encoder + predictor (Exp. 2, variant A)
      - JEPA Cluster: train_jepa_model(cluster sampler, epochs=JEPA_EPOCHS, seed=PROBE_SEED) (Exp. 2, variant D)
    Cached per (panel_key, horizon) so e.g. the medium/H=5 cell is shared by Tables 1 and 2.
    """
    if cache is not None and (panel_key, horizon) in cache:
        return cache[(panel_key, horizon)]

    X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
    X_ctx_te = data["X_ctx_test"]
    T, F_dim = X_ctx_tr.shape[1], X_ctx_tr.shape[2]
    id_pred = IdentityPredictor()

    proj = RandomLinearProjector(context_len=T, n_features=F_dim, out_dim=RANDOM_PROJ_OUT_DIM,
                                 seed=PROBE_SEED).to(device)
    Z_rand_tr = extract_representations(proj, id_pred, X_ctx_tr, device=device)
    Z_rand_te = extract_representations(proj, id_pred, X_ctx_te, device=device)

    set_seed(PROBE_SEED)
    untrained_enc = TimeSeriesEncoder(mag_preserving=True).to(device)
    untrained_pred = LatentPredictor(embed_dim=untrained_enc.out_dim, action_dim=0).to(device)
    Z_untr_tr = extract_representations(untrained_enc, untrained_pred, X_ctx_tr, device=device)
    Z_untr_te = extract_representations(untrained_enc, untrained_pred, X_ctx_te, device=device)

    jepa_enc, jepa_pred = train_jepa_model(
        X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=False,
        use_cluster_sampler=True, mag_preserving=True,
        epochs=JEPA_EPOCHS, seed=PROBE_SEED, device=device,
    )
    Z_jepa_tr = extract_representations(jepa_enc, jepa_pred, X_ctx_tr, device=device)
    Z_jepa_te = extract_representations(jepa_enc, jepa_pred, X_ctx_te, device=device)

    feats = {
        "Random Linear": (Z_rand_tr, Z_rand_te),
        "Untrained Conv+Transformer": (Z_untr_tr, Z_untr_te),
        "JEPA Cluster": (Z_jepa_tr, Z_jepa_te),
    }
    EXTRACTOR_MODELS[(panel_key, horizon)] = {
        "Random Linear": (proj, id_pred),
        "Untrained Conv+Transformer": (untrained_enc, untrained_pred),
        "JEPA Cluster": (jepa_enc, jepa_pred),
    }
    if cache is not None:
        cache[(panel_key, horizon)] = feats
    return feats


def _probe_inputs(Z_tr, Z_te, data):
    """Probes see [Z, a_t] (same protocol as the paper)."""
    return np.hstack([Z_tr, data["act_train"]]), np.hstack([Z_te, data["act_test"]])


# %%
# ============================================================================
# TABLE 1 | ABLATION A: RANDOM LINEAR vs UNTRAINED CONV+TRANSFORMER vs JEPA
# (original DGP, medium noise, H=5; Ridge and TabICL probes; effective rank on Z_test alone)
# ============================================================================

def run_ablation_a_random_baseline(panel, device: str = "cpu", cache: dict = None):
    print("=" * 108)
    print("TABLE 1 | ABLATION A: RANDOM LINEAR vs UNTRAINED CONV+TRANSFORMER vs JEPA CLUSTER")
    print(f"         (original DGP, medium process noise, H = 5, Ridge and {PROBE_LABEL} probes)")
    if not HAS_TABICL:
        print("         NOTE: tabicl is not installed -> 'rbf-fb' rows are the RBF random-features + Ridge")
        print("         fallback, NOT TabICL. Install tabicl and re-run before reporting TabICL numbers.")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=5, train_end_day=TRAIN_END_DAY)
    feats = build_extractor_features(data, "medium", 5, device=device, cache=cache)
    y_rv_tr, y_rv_te = data["y_rv_train"], data["y_rv_test"]
    s_hor_tr, s_hor_te = data["s_hor_train"], data["s_hor_test"]
    asset_te = data["asset_test"]
    stride = max(1, 5 // 2)  # Experiment-2 TabICL support stride convention

    header = (
        f"{'Extractor':<30} | {'Probe':<7} | {'Eff. Rank':>9} | "
        f"{'RV Pooled R2':>12} | {'RV Within R2':>12} | {'State Within R2':>15}"
    )
    print(header)
    print("-" * len(header))
    for name, (Z_tr, Z_te) in feats.items():
        erank = compute_effective_rank(Z_te)
        F_tr, F_te = _probe_inputs(Z_tr, Z_te, data)
        for probe in ["ridge", "tabicl"]:
            y_hat = fit_predict_probe(F_tr, y_rv_tr, F_te, probe_type=probe, support_stride=stride, seed=PROBE_SEED)
            s_hat = fit_predict_probe(F_tr, s_hor_tr, F_te, probe_type=probe, support_stride=stride, seed=PROBE_SEED)
            label = "ridge" if probe == "ridge" else PROBE_LABEL
            print(
                f"{name:<30} | {label:<7} | {erank:>9.2f} | "
                f"{r2_score(y_rv_te, y_hat):>12.4f} | {within_asset_r2(y_rv_te, y_hat, asset_te):>12.4f} | "
                f"{within_asset_r2(s_hor_te, s_hat, asset_te):>15.4f}"
            )
    print("-" * len(header))


# %%
# ============================================================================
# TABLE 2 | ABLATION B: 1st/2nd/3rd-ORDER LATENT STATE RECOVERY
# (original DGP; horizons {1,5,10,22} x noise {low,medium,high}; fixed-capacity Ridge probe)
# ============================================================================

def higher_order_terms(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order-2: all 6 unique quadratic terms of s = (l, m, p).
    Order-3: selected 7 cubic / interaction terms: l^3, m^3, p^3, l^2 m, l^2 p, m^2 p, l m p.
    """
    l, m, p = s[:, 0], s[:, 1], s[:, 2]
    order2 = np.column_stack([l * l, l * m, l * p, m * m, m * p, p * p]).astype(np.float32)
    order3 = np.column_stack([l ** 3, m ** 3, p ** 3, l * l * m, l * l * p, m * m * p, l * m * p]).astype(np.float32)
    return order2, order3


def run_ablation_b_order_recovery(panels, device: str = "cpu", cache: dict = None):
    print("\n" + "=" * 108)
    print("TABLE 2 | ABLATION B: 1st / 2nd / 3rd-ORDER LATENT STATE RECOVERY, RIDGE PROBE")
    print("         (original DGP; horizons H in {1, 5, 10, 22} x process noise {low, medium, high})")
    print("=" * 108)

    for noise_name in ["low", "medium", "high"]:
        print(f"\nProcess noise = {noise_name} {NOISE_LEVELS[noise_name]}")
        header = (
            f"{'H':>3} | {'Extractor':<30} | {'Eff. Rank':>9} | "
            f"{'Order-1 Within R2':>18} | {'Order-2 Within R2':>18} | {'Order-3 Within R2':>18}"
        )
        print(header)
        print("-" * len(header))

        for H in ABLATION_HORIZONS:
            data = build_horizon_dataset(panels[noise_name], context_len=CONTEXT_LEN, horizon=H,
                                         train_end_day=TRAIN_END_DAY)
            feats = build_extractor_features(data, noise_name, H, device=device, cache=cache)
            s_hor_tr, s_hor_te = data["s_hor_train"], data["s_hor_test"]
            q_tr, c_tr = higher_order_terms(s_hor_tr)
            q_te, c_te = higher_order_terms(s_hor_te)
            asset_te = data["asset_test"]

            for name, (Z_tr, Z_te) in feats.items():
                erank = compute_effective_rank(Z_te)
                F_tr, F_te = _probe_inputs(Z_tr, Z_te, data)
                r2_orders = []
                for y_tr, y_te in [(s_hor_tr, s_hor_te), (q_tr, q_te), (c_tr, c_te)]:
                    y_hat = fit_predict_probe(F_tr, y_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
                    r2_orders.append(within_asset_r2(y_te, y_hat, asset_te))
                print(
                    f"{H:>3} | {name:<30} | {erank:>9.2f} | "
                    f"{r2_orders[0]:>18.4f} | {r2_orders[1]:>18.4f} | {r2_orders[2]:>18.4f}"
                )
        print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT C: BELIEF-BASED ACTION SELECTION (belief-space planning)
# Ensemble mean = expected dynamics, ensemble spread = epistemic risk on log-vol (process
# noise is unknowable to the model). Safety-first removes actions with estimated
# P(dangerous) > tol, then maximizes the conditional Sharpe among survivors.
# The "Action-JEPA enc. + lin-a" rows use the SAME trained Action-JEPA encoders as the
# "Action-JEPA Ens." rows, but read out [z, a] linearly (like the untrained rows): if they
# match the untrained rows, the decision gap comes from the predictor's nonlinear action
# pathway, not from the learned representation.
# ============================================================================

def _fit_ridge_probe(Z_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42):
    scaler = StandardScaler()
    model = Ridge(alpha=10.0, random_state=seed)
    model.fit(scaler.fit_transform(Z_tr), y_tr)
    return scaler, model


def _linear_in_action_belief(enc, pred, X_ctx_tr, act_tr, y_tr, X_eval, A_rep, K, seed, device):
    """Ridge on [z, a] from (enc, pred) features; returns predicted next states (K, n, 3)."""
    Z_tr = extract_representations(enc, pred, X_ctx_tr, device=device)
    scaler, ridge = _fit_ridge_probe(np.hstack([Z_tr, act_tr]), y_tr, seed=seed)
    Z_eval = extract_representations(enc, pred, X_eval, device=device)
    F_rep = np.hstack([np.tile(Z_eval, (K, 1)), A_rep])
    return ridge.predict(scaler.transform(F_rep)).reshape(K, len(X_eval), 3)


def run_experiment_c_belief_planning(panel, world: World, device: str = "cpu",
                                     n_ensemble: int = N_ENSEMBLE, n_eval: int = 1500,
                                     risk_tolerance: float = 0.20):
    print("\n" + "=" * 108)
    print("TABLE 3 | EXPERIMENT C: BELIEF-BASED ACTION SELECTION — CAN THE LEARNED STATE MACHINE")
    print(f"         PICK BETTER ACTIONS? ({world.name} DGP, H = 1, medium noise, 27-action grid,")
    print(f"         {n_ensemble}-member ensembles, safety filter at P(dangerous) <= {risk_tolerance}; seed {PROBE_SEED})")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=1,
                                 train_end_day=TRAIN_END_DAY, transition_fn=world.g)
    X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
    s_det1_tr = data["s_det1_train"]
    X_ctx_te = data["X_ctx_test"]
    s_curr_te, asset_te = data["s_curr_test"], data["asset_test"]
    betas = panel["asset_betas"]

    # Dangerous log-vol level: 90th percentile of TRAINING-period states (no test information)
    tau = float(np.quantile(panel["states"][:, :TRAIN_END_DAY, 0], 0.90))

    rng = np.random.default_rng(7)
    idx = rng.choice(len(X_ctx_te), size=min(n_eval, len(X_ctx_te)), replace=False)
    X_eval, s_curr, beta_eval = X_ctx_te[idx], s_curr_te[idx], betas[asset_te[idx]]
    K, n = len(ACTION_GRID), len(X_eval)
    cols = np.arange(n)

    # Oracle: true next state per (action, context) via the known transition function
    s_next_true = world.g_vec(s_curr[None, :, :], ACTION_GRID[:, None, :], beta_eval[None, :])  # (K, n, 3)
    U_true = state_utility(s_next_true)
    danger_true = s_next_true[:, :, 0] > tau
    a_star = np.argmax(U_true, axis=0)

    def policy_stats(a_hat: np.ndarray) -> Tuple[float, float, float, float]:
        u = U_true[a_hat, cols]
        return (
            float(u.mean()),
            float((U_true[a_star, cols] - u).mean()),
            float(100.0 * np.mean(a_hat == a_star)),
            float(100.0 * np.mean(danger_true[a_hat, cols])),
        )

    X_rep = np.tile(X_eval, (K, 1, 1))
    A_rep = np.repeat(ACTION_GRID, n, axis=0)
    jepa_states, jepa_lin_states, untr_states = [], [], []
    for m_idx in range(n_ensemble):
        # Action-JEPA member: belief from P(z_t, a_t) + Ridge readout
        seed_m = _ens_seed(0, m_idx)
        enc_m, pred_m = train_jepa_model(
            X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=True,
            use_cluster_sampler=True, mag_preserving=True,
            epochs=JEPA_EPOCHS, seed=seed_m, device=device,
        )
        Z_tr_m = extract_representations(enc_m, pred_m, X_ctx_tr, act_tr, use_action=True, device=device)
        scaler_m, ridge_m = _fit_ridge_probe(Z_tr_m, s_det1_tr, seed=seed_m)
        Z_rep = extract_representations(enc_m, pred_m, X_rep, A_rep, use_action=True, device=device)
        jepa_states.append(ridge_m.predict(scaler_m.transform(Z_rep)).reshape(K, n, 3))
        # Control: same trained encoder, linear-in-action readout
        jepa_lin_states.append(_linear_in_action_belief(
            enc_m, IdentityPredictor(), X_ctx_tr, act_tr, s_det1_tr, X_eval, A_rep, K, seed_m, device))

        # Untrained member: random encoder + predictor, linear-in-action readout
        seed_u = _ens_seed(100, m_idx)
        set_seed(seed_u)
        enc_u = TimeSeriesEncoder(mag_preserving=True).to(device)
        pred_u = LatentPredictor(embed_dim=enc_u.out_dim, action_dim=0).to(device)
        untr_states.append(_linear_in_action_belief(
            enc_u, pred_u, X_ctx_tr, act_tr, s_det1_tr, X_eval, A_rep, K, _ens_seed(0, m_idx), device))

    def make_policies(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = states.mean(axis=0)                    # (K, n, 3) expected dynamics
        sigma_l = states[:, :, :, 0].std(axis=0)    # (K, n) epistemic risk on log-vol
        p_danger = 0.5 * erfc((tau - mu[:, :, 0]) / (sigma_l * np.sqrt(2.0) + 1e-12))
        U_mu = state_utility(mu)
        neutral = np.argmax(U_mu, axis=0)
        safe = p_danger <= risk_tolerance
        greedy = np.argmax(np.where(safe, U_mu, -np.inf), axis=0)
        no_safe = ~safe.any(axis=0)
        greedy[no_safe] = np.argmin(p_danger[:, no_safe], axis=0)
        return neutral, greedy

    rows = [("Oracle (true g)",) + policy_stats(a_star)]
    for name, states in [("Action-JEPA Ens.", np.stack(jepa_states)),
                         ("Action-JEPA enc. + lin-a", np.stack(jepa_lin_states)),
                         ("Untrained Ens.", np.stack(untr_states))]:
        neutral, greedy = make_policies(states)
        rows.append((f"{name} risk-neutral",) + policy_stats(neutral))
        rows.append((f"{name} safety-first",) + policy_stats(greedy))
    rng_pol = np.random.default_rng(11)
    rows.append(("Random action",) + policy_stats(rng_pol.integers(K, size=n)))
    rows.append(("Zero action (do nothing)",) + policy_stats(np.full(n, ZERO_ACTION)))

    s_star = s_next_true[a_star, cols]
    above_med_vol = s_star[:, 0] >= np.median(s_next_true[:, :, 0], axis=0)
    print(f"Danger threshold tau = {tau:.3f} (90th pct of training-period log-vol); utility = E[r']/vol'")
    print(f"Oracle diagnostics: chosen next drift m' < 0 in {100 * np.mean(s_star[:, 1] < 0):.1f}% of contexts "
          f"(there, Sharpe rewards HIGHER vol); oracle picks above-median next log-vol in "
          f"{100 * np.mean(above_med_vol):.1f}%")
    header = f"{'Policy':<38} | {'Mean True U':>11} | {'Mean Regret':>11} | {'% Optimal':>9} | {'% Dangerous':>11}"
    print(header)
    print("-" * len(header))
    for name, u, regret, hit, dang in rows:
        print(f"{name:<38} | {u:>11.5f} | {regret:>11.5f} | {hit:>9.1f} | {dang:>11.1f}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT D: PORTFOLIO ALLOCATION. Every strategy solves the same exact QP;
# the oracle plugs in true moments, so regret >= 0 by construction.
# ============================================================================

def _portfolio_inputs(panel, data, feats, H: int):
    """(unique_dates, {extractor: (mu_hat, var_hat)} per (date, asset), mu_true, var_true)."""
    s_hor_tr = data["s_hor_train"]
    y_rv_tr = data["y_rv_train"]
    asset_te, date_te = data["asset_test"], data["date_test"]
    unique_dates = np.unique(date_te)
    n_assets = panel["states"].shape[0]

    def pivot(values):
        M = np.zeros((len(unique_dates), n_assets))
        M[np.searchsorted(unique_dates, date_te), asset_te] = values
        return M

    moments = {}
    for name, (Z_tr, Z_te) in feats.items():
        F_tr, F_te = _probe_inputs(Z_tr, Z_te, data)
        s_hat = fit_predict_probe(F_tr, s_hor_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        rv_hat = fit_predict_probe(F_tr, y_rv_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        mu = pivot(0.0005 * H * s_hat[:, 1])                        # E[H-day return] from recovered drift
        var = pivot(H * (np.exp(rv_hat) / np.sqrt(252.0)) ** 2)     # H-day variance from predicted RV
        moments[name] = (mu, var)

    # True H-day moments from the realized future DGP states (sum of daily means / variances)
    states = panel["states"]
    mu_true = np.zeros((len(unique_dates), n_assets))
    var_true = np.zeros((len(unique_dates), n_assets))
    for j, t in enumerate(unique_dates):
        fut = states[:, t + 1:t + 1 + H, :]
        mu_true[j] = 0.0005 * fut[:, :, 1].sum(axis=1)
        # v3 multiplied this sum by H as well (H^2 x the daily variance); fixed here
        var_true[j] = ((np.exp(np.clip(fut[:, :, 0], -4.5, 1.0)) / np.sqrt(252.0)) ** 2).sum(axis=1)
    return unique_dates, moments, mu_true, var_true


def _qp_weights(mu: np.ndarray, var: np.ndarray, cap: float) -> np.ndarray:
    """
    Exact maximizer of w'mu - 0.5 w'diag(var) w  s.t.  sum(w) = 1, 0 <= w <= cap.
    The QP is separable: w_i = clip((mu_i - nu) / var_i, 0, cap), with the multiplier nu
    found by bisection so the weights sum to one (requires cap * n >= 1).
    """
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    mu = np.asarray(mu, dtype=np.float64)
    lo, hi = float(np.min(mu - cap * var)) - 1e-12, float(np.max(mu))   # sum(w(lo)) = n*cap >= 1, sum(w(hi)) = 0
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        if np.clip((mu - nu) / var, 0.0, cap).sum() > 1.0:
            lo = nu
        else:
            hi = nu
    w = np.clip((mu - lo) / var, 0.0, cap)
    return w / w.sum()


def _inverse_vol_weights(var: np.ndarray, cap: float = 0.25) -> np.ndarray:
    raw = 1.0 / (var + 1e-12)
    w = np.minimum(raw / raw.sum(), cap)
    return w / w.sum()


def run_experiment_d_portfolio(panel, world: World, device: str = "cpu", cache: dict = None,
                               H: int = 5, weight_cap: float = 0.25):
    print("\n" + "=" * 112)
    print("TABLE 4 | EXPERIMENT D: PORTFOLIO ALLOCATION — DOES THE RECOVERED STATE MACHINE IMPROVE")
    print(f"         ALLOCATION? ({world.name} DGP, H = {H}, medium noise, long-only MV QP, cap {weight_cap:.0%},")
    print(f"         rebalanced every {H} days; seed {PROBE_SEED})")
    print("=" * 112)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H,
                                 train_end_day=TRAIN_END_DAY, transition_fn=world.g)
    feats = build_extractor_features(data, _world_key(world), H, device=device, cache=cache)
    unique_dates, moments, mu_true, var_true = _portfolio_inputs(panel, data, feats, H)

    rets = panel["returns"]
    n_assets = rets.shape[0]
    reb_idx = np.arange(0, len(unique_dates), H)
    var_hist = np.zeros((len(unique_dates), n_assets))
    for j, t in enumerate(unique_dates):
        var_hist[j] = H * rets[:, max(0, t - 21):t + 1].var(axis=1)

    def run_strategy(weight_fn):
        exp_u, realized = [], []
        for j in reb_idx:
            t = unique_dates[j]
            w = weight_fn(j)
            exp_u.append(float(w @ mu_true[j] - 0.5 * np.sum(w ** 2 * var_true[j])))
            realized.append(float(w @ rets[:, t + 1:t + 1 + H].sum(axis=1)))
        exp_u, realized = np.array(exp_u), np.array(realized)
        sharpe = float(realized.mean() / (realized.std() + 1e-12) * np.sqrt(252.0 / H))
        return float(exp_u.mean()), float(realized.sum()), sharpe

    strategies = [("Oracle QP (true moments)", lambda j: _qp_weights(mu_true[j], var_true[j], weight_cap))]
    for name, (mu_hat, var_hat) in moments.items():
        strategies.append((f"{name} QP", lambda j, mu=mu_hat, var=var_hat: _qp_weights(mu[j], var[j], weight_cap)))
    strategies.append(("Inverse-Vol (trailing 22d)", lambda j: _inverse_vol_weights(var_hist[j], weight_cap)))
    strategies.append(("Equal-Weight (1/N)", lambda j: np.full(n_assets, 1.0 / n_assets)))

    results = [(name,) + run_strategy(fn) for name, fn in strategies]
    oracle_u = results[0][1]

    print("Objective: true MV utility w'mu - 0.5 w'diag(Var) w with moments from the realized future DGP states.")
    print("All QP strategies solve the same problem exactly; only the plugged-in moments differ.")
    header = (f"{'Portfolio':<30} | {'Mean MV Utility (true)':>22} | {'Regret vs Oracle':>16} | "
              f"{'Realized Total':>14} | {'Realized Sharpe':>15}")
    print(header)
    print("-" * len(header))
    for name, exp_u, tot_ret, sharpe in results:
        print(f"{name:<30} | {exp_u:>22.6f} | {oracle_u - exp_u:>16.6f} | {tot_ret:>14.4f} | {sharpe:>15.2f}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT E: LP DECISION-COMPLEXITY SCALING
#     max_w w'mu_hat  s.t.  sum(w)=1, 0 <= w <= 2/n
# (box cap 2/n => ceil(n/2) active positions; n=2 => pure argmax). Regret vs the LP on true returns.
# ============================================================================

def run_experiment_e_lp_scaling(panel, world: World, device: str = "cpu", cache: dict = None, H: int = 5,
                                sizes: Tuple[int, ...] = (2, 4, 8, 16, 20), n_subsets: int = 8):
    print("\n" + "=" * 112)
    print("TABLE 5 | EXPERIMENT E: LP DECISION-COMPLEXITY SCALING — DOES DECISION REGRET GROW WITH")
    print(f"         WEIGHT-VECTOR SIZE n? ({world.name} DGP; max w'mu_hat s.t. sum(w)=1, 0<=w<=2/n; "
          f"{n_subsets} seeded subsets per")
    print(f"         size, 1 at n = {N_ASSETS} (the full panel); regret vs. the LP on true expected returns)")
    print("=" * 112)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H,
                                 train_end_day=TRAIN_END_DAY, transition_fn=world.g)
    feats = build_extractor_features(data, _world_key(world), H, device=device, cache=cache)
    unique_dates, moments, mu_true, _ = _portfolio_inputs(panel, data, feats, H)
    n_assets = mu_true.shape[1]
    reb_idx = np.arange(0, len(unique_dates), H)

    def solve_lp(mu_vec: np.ndarray, cap: float) -> np.ndarray:
        res = linprog(c=-mu_vec, A_eq=np.ones((1, len(mu_vec))), b_eq=[1.0],
                      bounds=[(0.0, cap)] * len(mu_vec), method="highs")
        return res.x if res.success else np.full(len(mu_vec), 1.0 / len(mu_vec))

    header = (f"{'n':>3} | {'Extractor':<30} | {'Mean Regret (true)':>18} | "
              f"{'Mean True E[R]':>14} | {'Top-1 Hit %':>10}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        rng = np.random.default_rng(PROBE_SEED + n)
        n_sub = n_subsets if n < n_assets else 1
        subsets = [np.sort(rng.choice(n_assets, size=n, replace=False)) for _ in range(n_sub)]
        cap = 2.0 / n

        u_star = np.zeros((len(subsets), len(reb_idx)))
        star_arg = np.zeros((len(subsets), len(reb_idx)), dtype=np.int64)
        for si, sub in enumerate(subsets):
            for jj, j in enumerate(reb_idx):
                u_star[si, jj] = float(solve_lp(mu_true[j, sub], cap) @ mu_true[j, sub])
                star_arg[si, jj] = int(np.argmax(mu_true[j, sub]))

        for name, (mu_hat, _) in moments.items():
            regrets, utils, hits = [], [], []
            for si, sub in enumerate(subsets):
                for jj, j in enumerate(reb_idx):
                    u_hat = float(solve_lp(mu_hat[j, sub], cap) @ mu_true[j, sub])
                    regrets.append(u_star[si, jj] - u_hat)
                    utils.append(u_hat)
                    hits.append(100.0 * (int(np.argmax(mu_hat[j, sub])) == star_arg[si, jj]))
            print(f"{n:>3} | {name:<30} | {np.mean(regrets):>18.6f} | {np.mean(utils):>14.6f} | {np.mean(hits):>10.1f}")

        eq_regret = [u_star[si, jj] - float(mu_true[j, sub].mean())
                     for si, sub in enumerate(subsets) for jj, j in enumerate(reb_idx)]
        print(f"{n:>3} | {'Equal-Weight (reference)':<30} | {np.mean(eq_regret):>18.6f} | {'—':>14} | {'—':>10}")
        print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENTS F/G MACHINERY: per-asset value iteration on a discretized state grid.
# Q(s, a) = mean_k [ U(g(s, a, beta) + eps_k) + gamma V(nearest_cell(g(s, a, beta) + eps_k)) ]
# with a FIXED set of noise draws eps_k, shared across actions (common random numbers), so value
# iteration is a deterministic gamma-contraction of the sample-average MDP.
# ============================================================================

def _cell_index(S: np.ndarray, axes, n_grid) -> np.ndarray:
    """Flat index of the nearest grid cell (clipped to the grid) for states S (..., 3)."""
    idx = [
        np.clip(np.rint((S[..., k] - axes[k][0]) / (axes[k][1] - axes[k][0])), 0, n_grid[k] - 1).astype(np.int64)
        for k in range(3)
    ]
    return np.ravel_multi_index(idx, n_grid)


def _solve_mdp(panel, g_vec, gamma: float = 0.9, n_iter: int = VI_ITERS,
               n_grid: Tuple[int, int, int] = (11, 9, 9), n_samples: int = 32) -> dict:
    """One value function per asset (g depends on the asset's beta). Grid bounds from training states."""
    flat = panel["states"][:, :TRAIN_END_DAY].reshape(-1, 3)
    lo, hi = np.percentile(flat, 0.5, axis=0), np.percentile(flat, 99.5, axis=0)
    axes = [np.linspace(lo[k], hi[k], n_grid[k]) for k in range(3)]
    cells = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    K = len(ACTION_GRID)
    rng = np.random.default_rng(PROBE_SEED)
    noise = rng.normal(0.0, PROCESS_NOISE, size=(n_samples, len(cells), 1, 3))   # shared across actions

    V_all = []
    for beta in panel["asset_betas"]:
        S_next = g_vec(cells[None, :, None, :], ACTION_GRID[None, None, :, :], float(beta)) + noise
        R = state_utility(S_next)                      # (n_samples, C, K)
        nxt = _cell_index(S_next, axes, n_grid)
        V = np.zeros(len(cells))
        for _ in range(n_iter):
            V = (R + gamma * V[nxt]).mean(axis=0).max(axis=1)
        V_all.append(V)
    return {"axes": axes, "n_grid": n_grid, "V": np.stack(V_all),
            "betas": np.asarray(panel["asset_betas"], dtype=np.float64), "gamma": gamma}


def _q_values(S: np.ndarray, assets: np.ndarray, mdp: dict, g_vec, n_samples: int, seed: int,
              chunk: int = 2048) -> np.ndarray:
    """Monte Carlo Q(s, a) for each row (its own asset's beta and V); returns (n_rows, 27).
    The same noise draws are shared by all 27 actions of a row (common random numbers), so
    action comparisons -- and therefore pi* labels -- are far less noisy than independent draws."""
    rng = np.random.default_rng(seed)
    K = len(ACTION_GRID)
    out = np.empty((len(S), K))
    for i in range(0, len(S), chunk):
        s, a_ids = S[i:i + chunk], assets[i:i + chunk]
        b = mdp["betas"][a_ids]
        S_next = (g_vec(s[None, :, None, :], ACTION_GRID[None, None, :, :], b[None, :, None])
                  + rng.normal(0.0, PROCESS_NOISE, size=(n_samples, len(s), 1, 3)))   # shared across actions
        V_next = mdp["V"][a_ids[None, :, None], _cell_index(S_next, mdp["axes"], mdp["n_grid"])]
        out[i:i + chunk] = (state_utility(S_next) + mdp["gamma"] * V_next).mean(axis=0)
    return out


def _policy_targets(data: Dict[str, np.ndarray], split: str, mdp: dict, g_vec, n_samples: int, seed: int):
    """(V*, pi*, Q) per row of a split, from Q-values on the row's true current state."""
    Q = _q_values(data[f"s_curr_{split}"], data[f"asset_{split}"], mdp, g_vec, n_samples, seed)
    return Q.max(axis=1), Q.argmax(axis=1), Q


def _bellman_regret(Q: np.ndarray, a_hat: np.ndarray) -> float:
    return float(np.mean(Q.max(axis=1) - Q[np.arange(len(Q)), a_hat]))


# %%
# ============================================================================
# EXPERIMENT F: RL EMULATION — can the representation recover the optimal route of the
# MDP induced by g? Probe V* (Ridge) and pi* (27-way one-hot Ridge readout).
# ============================================================================

def run_experiment_f_rl_emulation(panel, world: World, device: str = "cpu", cache: dict = None,
                                  H: int = 5, gamma: float = 0.9):
    print("\n" + "=" * 108)
    print("TABLE 6 | EXPERIMENT F: RL EMULATION — CAN THE REPRESENTATION RECOVER THE OPTIMAL ROUTE OF")
    print(f"         THE MDP INDUCED BY g? ({world.name} DGP, 27-action grid, per-asset value iteration,")
    print(f"         gamma = {gamma}, reward = conditional Sharpe of the next state; Monte Carlo Bellman regret)")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H,
                                 train_end_day=TRAIN_END_DAY, transition_fn=world.g)
    feats = build_extractor_features(data, _world_key(world), H, device=device, cache=cache)
    if world.name not in _MDP_CACHE:
        _MDP_CACHE[world.name] = _solve_mdp(panel, world.g_vec, gamma=gamma)
    mdp = _MDP_CACHE[world.name]

    V_tr, pi_tr, _ = _policy_targets(data, "train", mdp, world.g_vec, n_samples=64, seed=PROBE_SEED + 1)
    V_te, pi_te, Q_te = _policy_targets(data, "test", mdp, world.g_vec, n_samples=64, seed=PROBE_SEED + 2)
    onehot = np.eye(len(ACTION_GRID), dtype=np.float32)
    asset_te = data["asset_test"]

    header = (f"{'Policy / Extractor':<30} | {'MC Bellman Regret':>17} | {'pi* Accuracy %':>14} | "
              f"{'V* Within R2':>12} | {'V* Pooled R2':>12}")
    print(header)
    print("-" * len(header))
    print(f"{'Oracle (value iteration)':<30} | {0.0:>17.4f} | {100.0:>14.1f} | {'—':>12} | {'—':>12}")
    for name, (Z_tr, Z_te) in feats.items():
        F_tr, F_te = _probe_inputs(Z_tr, Z_te, data)
        a_hat = np.argmax(fit_predict_probe(F_tr, onehot[pi_tr], F_te, probe_type="ridge", seed=PROBE_SEED), axis=1)
        v_hat = fit_predict_probe(F_tr, V_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        print(f"{name:<30} | {_bellman_regret(Q_te, a_hat):>17.4f} | {100.0 * np.mean(a_hat == pi_te):>14.1f} | "
              f"{within_asset_r2(V_te, v_hat, asset_te):>12.3f} | {r2_score(V_te, v_hat):>12.3f}")
    rand_a = np.random.default_rng(PROBE_SEED + 4).integers(len(ACTION_GRID), size=len(Q_te))
    zero_a = np.full(len(Q_te), ZERO_ACTION)
    print(f"{'Random action':<30} | {_bellman_regret(Q_te, rand_a):>17.4f} | "
          f"{100.0 * np.mean(rand_a == pi_te):>14.1f} | {'—':>12} | {'—':>12}")
    print(f"{'Zero action (do nothing)':<30} | {_bellman_regret(Q_te, zero_a):>17.4f} | "
          f"{100.0 * np.mean(zero_a == pi_te):>14.1f} | {'—':>12} | {'—':>12}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT G: DYNAMICS SHIFT — is optimal behavior saved when the world changes to
# g_shift = g + kappa * tanh(s - center) on (l, m): a BOUNDED (saturating) repulsion
# from the original central mass, so the shifted world stays stationary?
# Transfer rows: A-trained extractors and probes applied to B, with B's features normalized
# by A's training statistics. Refit rows: Random Linear / Untrained keep the SAME frozen
# extractor and only refit the probes on B; JEPA Cluster is retrained on B.
# ============================================================================

def run_experiment_g_dynamics_shift(panel_A, world: World, device: str = "cpu", cache: dict = None,
                                    H: int = 5, kappa: float = 0.10, gamma: float = 0.9):
    print("\n" + "=" * 108)
    print("TABLE 7 | EXPERIMENT G: DYNAMICS SHIFT — IS OPTIMAL BEHAVIOR SAVED WHEN THE TRANSITION FUNCTION")
    print(f"         STEERS STATES AWAY FROM THE ORIGINAL CENTRAL MASS? ({world.name} DGP; g_shift = g + bounded")
    print(f"         tanh repulsion, kappa = {kappa}, from the original (l, m) mean; regret under the shifted MDP)")
    print("=" * 108)

    center = panel_A["states"].reshape(-1, 3).mean(axis=0)
    repel = np.array([kappa, kappa, 0.0])

    def g_shift(s, a, beta):
        # Saturating repulsion: bounded force keeps the shifted world stationary. (Linear repulsion
        # compounds with g's persistence and diverges -> float32 overflow -> NaN features.)
        return world.g(s, a, beta) + (repel.astype(np.float32) * np.tanh(s - center)).astype(np.float32)

    def g_shift_vec(S, A, beta):
        return world.g_vec(S, A, beta) + repel * np.tanh(np.asarray(S, dtype=np.float64) - center)

    cfg = SimConfig(n_assets=N_ASSETS, n_days=N_DAYS, context_len=CONTEXT_LEN)
    panel_B = generate_panel_with_interventions(cfg, seed=PANEL_SEED, state_noise=NOISE_LEVELS["medium"],
                                                transition_fn=g_shift)
    kw = dict(context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    data_A = build_horizon_dataset(panel_A, transition_fn=world.g, **kw)
    data_B = build_horizon_dataset(panel_B, transition_fn=g_shift, **kw)                 # refit rows
    data_Bx = build_horizon_dataset(panel_B, transition_fn=g_shift,
                                    norm_stats=data_A["norm_stats"], **kw)               # transfer rows
    assert np.array_equal(data_B["date_test"], data_Bx["date_test"])

    feats_A = build_extractor_features(data_A, _world_key(world), H, device=device, cache=cache)
    models = EXTRACTOR_MODELS[(_world_key(world), H)]
    if world.name not in _MDP_CACHE:
        _MDP_CACHE[world.name] = _solve_mdp(panel_A, world.g_vec, gamma=gamma)
    mdp_A = _MDP_CACHE[world.name]
    mdp_B = _solve_mdp(panel_B, g_shift_vec, gamma=gamma)

    _, pi_A_tr, _ = _policy_targets(data_A, "train", mdp_A, world.g_vec, n_samples=64, seed=PROBE_SEED + 5)
    _, pi_B_tr, _ = _policy_targets(data_B, "train", mdp_B, g_shift_vec, n_samples=64, seed=PROBE_SEED + 6)
    _, pi_B_te, Q_B_te = _policy_targets(data_B, "test", mdp_B, g_shift_vec, n_samples=64, seed=PROBE_SEED + 7)

    onehot = np.eye(len(ACTION_GRID), dtype=np.float32)
    s_hor_A_tr, s_hor_B_tr, s_hor_B_te = data_A["s_hor_train"], data_B["s_hor_train"], data_B["s_hor_test"]
    asset_B_te = data_B["asset_test"]

    jepa_B_enc, jepa_B_pred = train_jepa_model(
        data_B["X_ctx_train"], data_B["X_fut_train"], data_B["act_train"], use_action_predictor=False,
        use_cluster_sampler=True, mag_preserving=True, epochs=JEPA_EPOCHS, seed=PROBE_SEED, device=device,
    )

    def rep(model, X):
        return extract_representations(model[0], model[1], X, device=device)

    def score(F_tr, pi_tr, s_tr, F_te):
        a_hat = np.argmax(fit_predict_probe(F_tr, onehot[pi_tr], F_te, probe_type="ridge", seed=PROBE_SEED), axis=1)
        s_hat = fit_predict_probe(F_tr, s_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        return _bellman_regret(Q_B_te, a_hat), 100.0 * np.mean(a_hat == pi_B_te), within_asset_r2(s_hor_B_te, s_hat, asset_B_te)

    header = f"{'Policy (B-world)':<40} | {'MC Bellman Regret':>17} | {'pi*_B Acc %':>11} | {'State R2':>9}"
    print(f"Shift center (original l, m mean): ({center[0]:.3f}, {center[1]:.3f}); kappa = {kappa}")
    print(header)
    print("-" * len(header))
    print(f"{'Oracle-B (value iteration on g_shift)':<40} | {0.0:>17.4f} | {100.0:>11.1f} | {'—':>9}")

    for name, (Z_A_tr, _) in feats_A.items():
        F_A_tr = np.hstack([Z_A_tr, data_A["act_train"]])
        F_Bx_te = np.hstack([rep(models[name], data_Bx["X_ctx_test"]), data_Bx["act_test"]])
        regret, acc, r2 = score(F_A_tr, pi_A_tr, s_hor_A_tr, F_Bx_te)
        print(f"{f'{name}  A->B transferred':<40} | {regret:>17.4f} | {acc:>11.1f} | {r2:>9.3f}")

        model_B = (jepa_B_enc, jepa_B_pred) if name == "JEPA Cluster" else models[name]
        label = "retrained on B" if name == "JEPA Cluster" else "probes refit on B"
        F_B_tr = np.hstack([rep(model_B, data_B["X_ctx_train"]), data_B["act_train"]])
        F_B_te = np.hstack([rep(model_B, data_B["X_ctx_test"]), data_B["act_test"]])
        regret, acc, r2 = score(F_B_tr, pi_B_tr, s_hor_B_tr, F_B_te)
        print(f"{f'{name}  {label}':<40} | {regret:>17.4f} | {acc:>11.1f} | {r2:>9.3f}")

    rand_a = np.random.default_rng(PROBE_SEED + 8).integers(len(ACTION_GRID), size=len(Q_B_te))
    zero_a = np.full(len(Q_B_te), ZERO_ACTION)
    print(f"{'Random action':<40} | {_bellman_regret(Q_B_te, rand_a):>17.4f} | "
          f"{100.0 * np.mean(rand_a == pi_B_te):>11.1f} | {'—':>9}")
    print(f"{'Zero action (do nothing)':<40} | {_bellman_regret(Q_B_te, zero_a):>17.4f} | "
          f"{100.0 * np.mean(zero_a == pi_B_te):>11.1f} | {'—':>9}")
    print("-" * len(header))


# %%
# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Determinism (v3 Experiment C drifted between GPU runs). The cuBLAS setting must be in
    # place before the first CUDA call; warn_only keeps ops without a deterministic kernel running.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = make_world(ARGS.decision_dgp)
    print(f"Running Study 3 ablations on device={device} | seed={PROBE_SEED} | decision DGP={world.name} | "
          f"probe={'TabICL' if HAS_TABICL else 'RBF fallback (tabicl NOT installed)'}"
          f"{' | QUICK SMOKE TEST' if ARGS.quick else ''}")
    cfg = SimConfig(n_assets=N_ASSETS, n_days=N_DAYS, context_len=CONTEXT_LEN)

    # Ablations A/B: the original DGP at three noise levels (same panels as Experiments 1-2)
    panels = {
        noise_name: generate_panel_with_interventions(cfg, seed=PANEL_SEED, state_noise=noise)
        for noise_name, noise in NOISE_LEVELS.items()
    }
    # Experiments C-G: the decision DGP at medium noise
    decision_panel = panels["medium"] if world.name == "original" else generate_panel_with_interventions(
        cfg, seed=PANEL_SEED, state_noise=NOISE_LEVELS["medium"], transition_fn=world.g)

    print("DGP diagnostics (medium noise):")
    print_dgp_diagnostics("original", panels["medium"])
    if world.name != "original":
        print_dgp_diagnostics(world.name, decision_panel)

    feature_cache = {}
    run_ablation_a_random_baseline(panels["medium"], device=device, cache=feature_cache)
    run_ablation_b_order_recovery(panels, device=device, cache=feature_cache)
    run_experiment_c_belief_planning(decision_panel, world, device=device)
    run_experiment_d_portfolio(decision_panel, world, device=device, cache=feature_cache)
    run_experiment_e_lp_scaling(decision_panel, world, device=device, cache=feature_cache)
    run_experiment_f_rl_emulation(decision_panel, world, device=device, cache=feature_cache)
    run_experiment_g_dynamics_shift(decision_panel, world, device=device, cache=feature_cache)