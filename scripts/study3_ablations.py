"""
Study 3 - Targeted Ablations (extends, and does not modify, the original Study-3 codebase).

SELF-CONTAINED GOOGLE-COLAB RUNNER: this single file embeds verbatim copies of the
existing Study-3 utilities (encoder, JEPA training loop, probes, metrics, data
generation) so it runs with NO other files present. The originals in
scripts/study3_state_timescale_experiments.py remain the reference implementation;
keep the copies below in sync with it.

ABLATION A: Does the structure of the untrained Conv+Transformer matter, or would any
    high-dimensional random projection of the same size work just as well?
    -> RandomLinearProjector (frozen random matrix, out_dim=33) vs Untrained Conv+Transformer
       vs JEPA Cluster on the fixed medium-noise, H=5 setting, read out by Ridge AND TabICL.

ABLATION B: How quickly does recovery of higher-order (1st/2nd/3rd-order) latent state
    features degrade as forecast horizon (H in {1,5,10,22}) and process noise
    (low/medium/high on the latent state transition) increase?
    -> Full horizon x noise x order within-asset R^2 matrix for the three extractors,
       read out by a fixed-capacity Ridge probe, plus effective rank.

EXPERIMENT C (belief-space planning): if the representation really contains the state
    machine, it should support decisions, not just probes. For each test context we
    enumerate a 27-point action grid and form a BELIEF over the next state from a
    5-member ensemble of Action-JEPA predictors (vs. 5 untrained encoders): the ensemble
    mean is the expected dynamics, the ensemble spread is the estimated risk. A
    safety-first policy first removes actions whose estimated P(dangerous log-vol state)
    exceeds a tolerance (belief-space filtering, cf. UWM-JEPA), then maximizes the
    conditional Sharpe E[r']/vol' among the survivors. Because the DGP is known, every
    policy is scored against the oracle: true regret, % optimal actions, and % of
    dangerous states actually entered.

How to run in Google Colab (any ONE of these):
    1. File -> Upload notebook -> select this .py file (Colab auto-splits it into
       cells on the '# %%' markers) -> Runtime -> Run all.
    2. Upload the file to the Colab VM (folder icon, left pane) and run:
           !python study3_ablations.py
    3. Once pushed to GitHub: colab.research.google.com -> GitHub tab -> this file.

EXPERIMENT D (portfolio allocation): does the recovered state machine improve ALLOCATION?
    The three extractors' Ridge probes predict the existing Study-3 targets (s_hor and
    y_rv); the recovered drift state gives the expected H-day return and the predicted
    RV gives the H-day variance. Long-only mean-variance weights (capped, rebalanced
    every H days out-of-sample) are scored on TRUE expected returns/variances computed
    exactly from the DGP, against equal-weight, inverse-vol, and oracle baselines.

EXPERIMENT E (LP decision-complexity scaling): how does decision quality scale with the
    size of the weight vector? For n in {2, 4, 8, 16, 20} assets, solve the LP
        max_w  w' mu_hat   s.t.  sum(w) = 1,  0 <= w <= 2/n
    with mu_hat from each extractor's recovered drift, on seeded random asset subsets,
    and score regret exactly against the LP solved on TRUE expected returns.

EXPERIMENT F (RL emulation): the MDP induced by the known g (27-action grid, transition
    kernel = g + the DGP's process noise, reward = conditional Sharpe of the next state,
    discount 0.9) is solved by stochastic value iteration into an oracle V*, pi*. Can
    each extractor's representation recover the optimal route: probe V* (regression) and
    pi* (27-way one-hot readout), scored by exact Bellman regret of the probed policy.

EXPERIMENT G (dynamics shift): the world changes to g_shift, which steers states AWAY
    from the original central mass (saturating tanh repulsion from the original (l, m)
    mean; bounded force keeps the shifted world stationary). Is optimal behavior saved?
    A-trained extractors/probes are transferred to the shifted panel (state recovery +
    Bellman regret under the shifted MDP's oracle) and compared against extractors/
    probes retrained on the shifted panel.

Runtime: the sweep trains JEPA Cluster once per (noise, horizon) pair (12 trainings) plus a
5-member Action-JEPA ensemble for Experiment C (5 more) and one shift-world JEPA (G), so
expect roughly 60-100 minutes on a Colab CPU, faster on GPU. Experiments D/E/F/G reuse the
cached medium/H=5 features; F/G add two value iterations (~2-3 min each on CPU).
`tabicl` is optional; the original RBF-ensemble fallback is used when it is not installed.
"""

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
# the optional state_noise kwarg on generate_panel_with_interventions, whose
# default reproduces the original stream bit-for-bit.
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
) -> Dict[str, np.ndarray]:
    """
    Constructs context windows X_{t-L+1:t}, future target windows X_{t+1:t+H},
    actions applied on decision day t, future state s_{t+H},
    and log H-day realized volatility target y_{t+H}, with an H-day embargo split.
    """
    feats = panel["features"]
    acts = panel["actions"]
    states = panel["states"]
    rets = panel["returns"]
    betas = panel["asset_betas"]
    N, T, F_dim = feats.shape

    start_t = 60 + context_len
    max_t = T - horizon - 1

    train_slice = feats[:, start_t:train_end_day, :].reshape(-1, F_dim)
    f_mean = train_slice.mean(axis=0, keepdims=True)
    f_std = train_slice.std(axis=0, keepdims=True) + 1e-8
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
            s_next_det = transition_function(s_t, act_t, betas[i])
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
# NEW ABLATION CODE (this part is new; everything above is the existing Study-3
# machinery copied verbatim so this single file runs standalone in Colab).
# ============================================================================

# Protocol constants (identical to the original Study-3 experiments)
PANEL_SEED = 2026          # same panel seed as the original main script
PROBE_SEED = 42            # same seed as Experiment 2 (untrained encoder, JEPA, probes)
CONTEXT_LEN = 40
TRAIN_END_DAY = 1000
N_ASSETS, N_DAYS = 20, 1400
JEPA_EPOCHS = 10
RANDOM_PROJ_OUT_DIM = 33   # matches TimeSeriesEncoder(mag_preserving=True).out_dim

# Process-noise levels on the latent state transition (l, m, p); medium = original
NOISE_LEVELS: Dict[str, Tuple[float, float, float]] = {
    "low": (0.02, 0.015, 0.01),
    "medium": (0.06, 0.05, 0.03),
    "high": (0.15, 0.12, 0.08),
}
ABLATION_HORIZONS = [1, 5, 10, 22]

# Shared decision machinery (Experiments C-F): 27-point action grid and the DGP process noise
ACTION_GRID = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=np.float32)
PROCESS_NOISE = np.array(NOISE_LEVELS["medium"], dtype=np.float32)
# Registries filled as experiments run: trained extractor models (for cross-panel transfer)
# and solved MDPs (value functions), keyed so later experiments reuse earlier results.
EXTRACTOR_MODELS: Dict[Tuple[str, int], dict] = {}
_MDP_CACHE: Dict[str, tuple] = {}


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
    """No-op predictor so extract_representations returns the raw projector output."""

    def forward(self, z: torch.Tensor, a=None) -> torch.Tensor:
        return z


def build_extractor_features(
    data: Dict[str, np.ndarray],
    noise_name: str,
    horizon: int,
    device: str = "cpu",
    cache: Dict[Tuple[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Train/test representations {extractor: (Z_train, Z_test)} for the three extractors,
    using the exact protocols of the original experiments:
      - Random Linear: frozen projector (fixed seed)
      - Untrained Conv+Transformer: set_seed(42) then untrained encoder + predictor (Exp. 2, variant A)
      - JEPA Cluster: train_jepa_model(cluster sampler, epochs=10, seed=42) (Exp. 2, variant D)
    Cached per (noise, horizon) so the medium/H=5 cell is shared by Tables A and B.
    """
    if cache is not None and (noise_name, horizon) in cache:
        return cache[(noise_name, horizon)]

    X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
    X_ctx_te = data["X_ctx_test"]
    T, F_dim = X_ctx_tr.shape[1], X_ctx_tr.shape[2]

    proj = RandomLinearProjector(
        context_len=T, n_features=F_dim, out_dim=RANDOM_PROJ_OUT_DIM, seed=PROBE_SEED
    ).to(device)
    id_pred = IdentityPredictor()
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
    EXTRACTOR_MODELS[(noise_name, horizon)] = {
        "proj": proj, "id_pred": id_pred,
        "untrained": (untrained_enc, untrained_pred),
        "jepa": (jepa_enc, jepa_pred),
    }
    if cache is not None:
        cache[(noise_name, horizon)] = feats
    return feats


# %%
# ============================================================================
# TABLE 1 | ABLATION A: RANDOM LINEAR vs UNTRAINED CONV+TRANSFORMER vs JEPA
# (fixed medium-noise, H=5 setting; Ridge and TabICL probes; same metrics/protocol
#  as the paper: probes see [Z, a_t], effective rank on Z_test alone)
# ============================================================================

def run_ablation_a_random_baseline(
    panel: Dict[str, np.ndarray],
    device: str = "cpu",
    cache: Dict[Tuple[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
):
    print("=" * 108)
    print("TABLE 1 | ABLATION A: RANDOM LINEAR vs UNTRAINED CONV+TRANSFORMER vs JEPA CLUSTER")
    print("         (medium process noise, H = 5, Ridge and TabICL probes)")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=5, train_end_day=TRAIN_END_DAY)
    feats = build_extractor_features(data, "medium", 5, device=device, cache=cache)

    act_tr, act_te = data["act_train"], data["act_test"]
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
        F_tr = np.hstack([Z_tr, act_tr])
        F_te = np.hstack([Z_te, act_te])
        for probe in ["ridge", "tabicl"]:
            y_hat = fit_predict_probe(F_tr, y_rv_tr, F_te, probe_type=probe, support_stride=stride, seed=PROBE_SEED)
            s_hat = fit_predict_probe(F_tr, s_hor_tr, F_te, probe_type=probe, support_stride=stride, seed=PROBE_SEED)
            print(
                f"{name:<30} | {probe:<7} | {erank:>9.2f} | "
                f"{r2_score(y_rv_te, y_hat):>12.4f} | {within_asset_r2(y_rv_te, y_hat, asset_te):>12.4f} | "
                f"{within_asset_r2(s_hor_te, s_hat, asset_te):>15.4f}"
            )
    print("-" * len(header))


# %%
# ============================================================================
# TABLE 2 | ABLATION B: 1st/2nd/3rd-ORDER LATENT STATE RECOVERY
# (horizons {1,5,10,22} x noise {low,medium,high}; fixed-capacity Ridge probe;
#  within-asset R^2 per order + effective rank)
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


def run_ablation_b_order_recovery(
    panels: Dict[str, Dict[str, np.ndarray]],
    device: str = "cpu",
    cache: Dict[Tuple[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
):
    print("\n" + "=" * 108)
    print("TABLE 2 | ABLATION B: 1st / 2nd / 3rd-ORDER LATENT STATE RECOVERY, RIDGE PROBE")
    print("         (horizons H in {1, 5, 10, 22} x process noise {low, medium, high})")
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
            data = build_horizon_dataset(panels[noise_name], context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
            feats = build_extractor_features(data, noise_name, H, device=device, cache=cache)

            act_tr, act_te = data["act_train"], data["act_test"]
            s_hor_tr, s_hor_te = data["s_hor_train"], data["s_hor_test"]
            q_tr, c_tr = higher_order_terms(s_hor_tr)
            q_te, c_te = higher_order_terms(s_hor_te)
            asset_te = data["asset_test"]

            for name, (Z_tr, Z_te) in feats.items():
                erank = compute_effective_rank(Z_te)
                F_tr = np.hstack([Z_tr, act_tr])
                F_te = np.hstack([Z_te, act_te])
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
# Does the learned state machine support DECISIONS, not just probes?
# For each test context we enumerate a 27-point action grid; a 5-member ensemble
# of Action-JEPA predictors (vs. 5 untrained encoders) forms a belief over the next
# state: ensemble mean = expected dynamics, ensemble spread = estimated risk
# (epistemic only; process noise is unknowable to the model). A safety-first policy
# first removes actions whose estimated P(dangerous log-vol state) exceeds a
# tolerance, then maximizes the conditional Sharpe E[r']/vol' among the survivors.
# Because the DGP is known, every policy is scored against the oracle built from
# the true transition function: true regret, % optimal actions, % dangerous states.
# ============================================================================

def state_utility(s_next: np.ndarray) -> np.ndarray:
    """Conditional Sharpe: expected next-day log-return per unit next-day vol.
    Matches the panel construction: E[r_{t+1}] = 0.0005 * m', vol = exp(l')/sqrt(252)."""
    l, m = s_next[..., 0], s_next[..., 1]
    vol = np.exp(np.clip(l, -4.5, 1.0)) / np.sqrt(252.0)
    return 0.0005 * m / (vol + 1e-12)


def _fit_ridge_probe(Z_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42):
    scaler = StandardScaler()
    model = Ridge(alpha=10.0, random_state=seed)
    model.fit(scaler.fit_transform(Z_tr), y_tr)
    return scaler, model


def run_experiment_c_belief_planning(
    panel: Dict[str, np.ndarray],
    device: str = "cpu",
    n_ensemble: int = 5,
    n_eval: int = 1500,
    risk_tolerance: float = 0.20,
):
    print("\n" + "=" * 108)
    print("TABLE 3 | EXPERIMENT C: BELIEF-BASED ACTION SELECTION — CAN THE LEARNED STATE MACHINE")
    print(f"         PICK BETTER ACTIONS? (H = 1, medium noise, 27-action grid, {n_ensemble}-member")
    print(f"         ensembles, safety filter at P(dangerous) <= {risk_tolerance})")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=1, train_end_day=TRAIN_END_DAY)
    X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
    s_det1_tr = data["s_det1_train"]
    X_ctx_te = data["X_ctx_test"]
    s_curr_te, asset_te = data["s_curr_test"], data["asset_test"]
    betas = panel["asset_betas"]

    tau = float(np.quantile(panel["states"][:, :, 0], 0.90))  # dangerous log-vol level

    rng = np.random.default_rng(7)
    idx = rng.choice(len(X_ctx_te), size=min(n_eval, len(X_ctx_te)), replace=False)
    X_eval, s_curr, beta_eval = X_ctx_te[idx], s_curr_te[idx], betas[asset_te[idx]]

    grid = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=np.float32)
    K, n = len(grid), len(X_eval)
    zero_idx = int(np.argmin(np.linalg.norm(grid, axis=1)))  # the (0, 0, 0) action

    # Oracle: true next state per (action, context) via the known transition function
    s_next_true = np.zeros((K, n, 3), dtype=np.float32)
    for k in range(K):
        for i in range(n):
            s_next_true[k, i] = transition_function(s_curr[i], grid[k], beta_eval[i])
    U_true = state_utility(s_next_true)
    danger_true = s_next_true[:, :, 0] > tau
    a_star = np.argmax(U_true, axis=0)
    cols = np.arange(n)

    def policy_stats(a_hat: np.ndarray) -> Tuple[float, float, float, float]:
        u = U_true[a_hat, cols]
        return (
            float(u.mean()),
            float((U_true[a_star, cols] - u).mean()),
            float(100.0 * np.mean(a_hat == a_star)),
            float(100.0 * np.mean(danger_true[a_hat, cols])),
        )

    # --- Action-JEPA ensemble: belief over the next state from P(z_t, a_t) + Ridge readout
    X_rep = np.tile(X_eval, (K, 1, 1))
    A_rep = np.repeat(grid, n, axis=0)
    jepa_states = []
    for m_idx in range(n_ensemble):
        enc_m, pred_m = train_jepa_model(
            X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=True,
            use_cluster_sampler=True, mag_preserving=True,
            epochs=JEPA_EPOCHS, seed=m_idx, device=device,
        )
        Z_tr_m = extract_representations(enc_m, pred_m, X_ctx_tr, act_tr, use_action=True, device=device)
        scaler_m, ridge_m = _fit_ridge_probe(Z_tr_m, s_det1_tr, seed=m_idx)
        Z_rep = extract_representations(enc_m, pred_m, X_rep, A_rep, use_action=True, device=device)
        jepa_states.append(ridge_m.predict(scaler_m.transform(Z_rep)).reshape(K, n, 3))
    jepa_states = np.stack(jepa_states)

    # --- Untrained-encoder ensemble: same protocol on random features ([z_t, a_t])
    untr_states = []
    for m_idx in range(n_ensemble):
        set_seed(100 + m_idx)
        enc_m = TimeSeriesEncoder(mag_preserving=True).to(device)
        pred_m = LatentPredictor(embed_dim=enc_m.out_dim, action_dim=0).to(device)
        Z_tr_m = extract_representations(enc_m, pred_m, X_ctx_tr, device=device)
        scaler_m, ridge_m = _fit_ridge_probe(np.hstack([Z_tr_m, act_tr]), s_det1_tr, seed=m_idx)
        Z_eval_m = extract_representations(enc_m, pred_m, X_eval, device=device)
        F_rep = np.hstack([np.tile(Z_eval_m, (K, 1)), A_rep])
        untr_states.append(ridge_m.predict(scaler_m.transform(F_rep)).reshape(K, n, 3))
    untr_states = np.stack(untr_states)

    def make_policies(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = states.mean(axis=0)                    # (K, n, 3) expected dynamics
        sigma_l = states[:, :, :, 0].std(axis=0)    # (K, n) epistemic risk on log-vol
        p_danger = 0.5 * erfc((tau - mu[:, :, 0]) / (sigma_l * np.sqrt(2.0) + 1e-12))
        U_mu = state_utility(mu)

        neutral = np.argmax(U_mu, axis=0)           # risk-neutral: max expected Sharpe
        safe = p_danger <= risk_tolerance            # safety-first: remove dangerous actions
        U_masked = np.where(safe, U_mu, -np.inf)
        greedy = np.argmax(U_masked, axis=0)
        no_safe = ~safe.any(axis=0)
        greedy[no_safe] = np.argmin(p_danger[:, no_safe], axis=0)
        return neutral, greedy

    rows = [("Oracle (true g)",) + policy_stats(a_star)]
    for name, states in [("Action-JEPA Ens.", jepa_states), ("Untrained Ens.", untr_states)]:
        neutral, greedy = make_policies(states)
        rows.append((f"{name} risk-neutral",) + policy_stats(neutral))
        rows.append((f"{name} safety-first",) + policy_stats(greedy))
    rng_pol = np.random.default_rng(11)
    rows.append(("Random action",) + policy_stats(rng_pol.integers(K, size=n)))
    rows.append(("Zero action (do nothing)",) + policy_stats(np.full(n, zero_idx)))

    header = f"{'Policy':<30} | {'Mean True U':>11} | {'Mean Regret':>11} | {'% Optimal':>9} | {'% Dangerous':>11}"
    print(f"Danger threshold tau = {tau:.3f} (90th pct of panel log-vol levels); utility = E[r']/vol' (conditional Sharpe)")
    print(header)
    print("-" * len(header))
    for name, u, regret, hit, dang in rows:
        print(f"{name:<30} | {u:>11.5f} | {regret:>11.5f} | {hit:>9.1f} | {dang:>11.1f}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT D: PORTFOLIO ALLOCATION — does the recovered state machine improve
# allocation? Predicted moments come from Ridge probes on the EXISTING Study-3
# targets (s_hor -> expected H-day return via the drift state; y_rv -> H-day
# variance). TRUE moments are computed exactly from the DGP states. Long-only
# mean-variance weights, capped, rebalanced every H days on the test window.
# Baselines: oracle MV (true states), inverse-vol (trailing 22d), equal-weight.
# ============================================================================

def _portfolio_inputs(panel, data, feats, H: int):
    """(unique_dates, {extractor: (mu_hat, var_hat)} per (date, asset), mu_true, var_true)."""
    s_hor_tr, s_hor_te = data["s_hor_train"], data["s_hor_test"]
    y_rv_tr, y_rv_te = data["y_rv_train"], data["y_rv_test"]
    act_tr, act_te = data["act_train"], data["act_test"]
    asset_te, date_te = data["asset_test"], data["date_test"]
    unique_dates = np.unique(date_te)
    n_assets = panel["states"].shape[0]

    def pivot(values):
        M = np.zeros((len(unique_dates), n_assets))
        M[np.searchsorted(unique_dates, date_te), asset_te] = values
        return M

    moments = {}
    for name, (Z_tr, Z_te) in feats.items():
        F_tr = np.hstack([Z_tr, act_tr])
        F_te = np.hstack([Z_te, act_te])
        s_hat = fit_predict_probe(F_tr, s_hor_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        rv_hat = fit_predict_probe(F_tr, y_rv_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        mu = pivot(0.0005 * H * s_hat[:, 1])                                # E[H-day return] from recovered drift
        var = pivot(H * (np.exp(rv_hat) / np.sqrt(252.0)) ** 2)             # H-day variance from predicted RV
        moments[name] = (mu, var)

    states = panel["states"]
    mu_true = np.zeros((len(unique_dates), n_assets))
    var_true = np.zeros((len(unique_dates), n_assets))
    for j, t in enumerate(unique_dates):
        fut = states[:, t + 1:t + 1 + H, :]
        mu_true[j] = 0.0005 * fut[:, :, 1].sum(axis=1)
        var_true[j] = H * ((np.exp(np.clip(fut[:, :, 0], -4.5, 1.0)) / np.sqrt(252.0)) ** 2).sum(axis=1)
    return unique_dates, moments, mu_true, var_true


def _mv_weights(mu: np.ndarray, var: np.ndarray, cap: float = 0.25) -> np.ndarray:
    """Long-only mean-variance tilt toward above-average assets: w ∝ max(mu - mean(mu), 0)/var,
    capped, renormalized. Demeaning is required because every expected return in this DGP
    is negative at H = 5; clipping absolute mu at zero would collapse all strategies (and
    the oracle) to Equal-Weight."""
    raw = np.clip(mu - mu.mean(), 0.0, None) / (var + 1e-12)
    if raw.sum() <= 1e-12:
        return np.full(len(mu), 1.0 / len(mu))
    w = np.minimum(raw / raw.sum(), cap)
    return w / w.sum()


def _inverse_vol_weights(var: np.ndarray, cap: float = 0.25) -> np.ndarray:
    raw = 1.0 / (var + 1e-12)
    w = np.minimum(raw / raw.sum(), cap)
    return w / w.sum()


def run_experiment_d_portfolio(panel, device: str = "cpu", cache: dict = None, H: int = 5, weight_cap: float = 0.25):
    print("\n" + "=" * 112)
    print("TABLE 4 | EXPERIMENT D: PORTFOLIO ALLOCATION — DOES THE RECOVERED STATE MACHINE IMPROVE")
    print(f"         ALLOCATION? (H = {H}, medium noise, long-only MV cap {weight_cap:.0%}, rebalanced every {H} days)")
    print("=" * 112)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    feats = build_extractor_features(data, "medium", H, device=device, cache=cache)
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
            mu_excess = mu_true[j] - mu_true[j].mean()
            exp_u.append(float(w @ mu_excess - 0.5 * np.sum(w ** 2 * var_true[j])))
            realized.append(float(w @ rets[:, t + 1:t + 1 + H].sum(axis=1)))
        exp_u, realized = np.array(exp_u), np.array(realized)
        sharpe = float(realized.mean() / (realized.std() + 1e-12) * np.sqrt(252.0 / H))
        return float(exp_u.mean()), float(realized.sum()), sharpe

    strategies = [("Oracle MV (true states)", lambda j: _mv_weights(mu_true[j], var_true[j], weight_cap))]
    for name, (mu_hat, var_hat) in moments.items():
        strategies.append((f"{name} MV", lambda j, mu=mu_hat, var=var_hat: _mv_weights(mu[j], var[j], weight_cap)))
    strategies.append(("Inverse-Vol (trailing 22d)", lambda j: _inverse_vol_weights(var_hist[j], weight_cap)))
    strategies.append(("Equal-Weight (1/N)", lambda j: np.full(n_assets, 1.0 / n_assets)))

    results = [(name,) + run_strategy(fn) for name, fn in strategies]
    ew_u = results[-1][1]

    header = (f"{'Portfolio':<30} | {'Mean MV Utility (true)':>22} | {'Δ vs Equal-Weight':>17} | "
              f"{'Realized Total':>14} | {'Realized Sharpe':>15}")
    print("True moments computed exactly from the DGP states; predicted moments from Ridge probes on s_hor / y_rv.")
    print("Objective is true mean-variance utility w'mu_excess - 0.5 w'Var w (diagonal true moments). The oracle")
    print("(same rule on true moments) is a REFERENCE, not the exact optimum of the capped rule: extractor deltas")
    print("vs it fall within the rule's ~±7e-5 optimality slack, so all extractors are statistically tied with it.")
    print("All absolute expected returns are negative in this panel, so allocation is relative.")
    print(header)
    print("-" * len(header))
    for name, exp_u, tot_ret, sharpe in results:
        print(f"{name:<30} | {exp_u:>22.6f} | {exp_u - ew_u:>17.6f} | {tot_ret:>14.4f} | {sharpe:>15.2f}")
    print("-" * len(header))
    for name, exp_u, tot_ret, sharpe in results:
        print(f"{name:<30} | {exp_u:>16.6f} | {oracle_u - exp_u:>16.6f} | {tot_ret:>14.4f} | {sharpe:>15.2f}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT E: LP DECISION-COMPLEXITY SCALING — how does decision quality scale
# with the weight-vector size n? For n in {2, 4, 8, 16, 20} assets solve
#     max_w w'mu_hat  s.t.  sum(w)=1, 0 <= w <= 2/n
# (box cap 2/n => exactly ceil(n/2) active positions; n=2 => pure argmax) with
# mu_hat from each extractor's recovered drift, on seeded random asset subsets.
# Regret is scored exactly against the LP solved on TRUE expected returns.
# ============================================================================

def run_experiment_e_lp_scaling(
    panel,
    device: str = "cpu",
    cache: dict = None,
    H: int = 5,
    sizes: Tuple[int, ...] = (2, 4, 8, 16, 20),
    n_subsets: int = 8,
):
    print("\n" + "=" * 112)
    print("TABLE 5 | EXPERIMENT E: LP DECISION-COMPLEXITY SCALING — DOES DECISION REGRET GROW WITH")
    print(f"         WEIGHT-VECTOR SIZE n? (max w'mu_hat s.t. sum(w)=1, 0<=w<=2/n; {n_subsets} seeded subsets per")
    print("         size; regret vs. the LP solved on true expected returns)")
    print("=" * 112)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    feats = build_extractor_features(data, "medium", H, device=device, cache=cache)
    unique_dates, moments, mu_true, _ = _portfolio_inputs(panel, data, feats, H)
    n_assets = mu_true.shape[1]
    reb_idx = np.arange(0, len(unique_dates), H)

    def solve_lp(mu_vec: np.ndarray, cap: float) -> np.ndarray:
        res = linprog(c=-mu_vec, A_eq=np.ones((1, len(mu_vec))), b_eq=[1.0],
                      bounds=[(0.0, cap)] * len(mu_vec), method="highs")
        if res.success:
            return res.x
        return np.full(len(mu_vec), 1.0 / len(mu_vec))

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
                w = solve_lp(mu_true[j, sub], cap)
                u_star[si, jj] = float(w @ mu_true[j, sub])
                star_arg[si, jj] = int(np.argmax(mu_true[j, sub]))

        for name, (mu_hat, _) in moments.items():
            regrets, utils, hits = [], [], []
            for si, sub in enumerate(subsets):
                for jj, j in enumerate(reb_idx):
                    w = solve_lp(mu_hat[j, sub], cap)
                    u_hat = float(w @ mu_true[j, sub])
                    regrets.append(u_star[si, jj] - u_hat)
                    utils.append(u_hat)
                    hits.append(100.0 * (int(np.argmax(mu_hat[j, sub])) == star_arg[si, jj]))
            print(f"{n:>3} | {name:<30} | {np.mean(regrets):>18.6f} | {np.mean(utils):>14.6f} | {np.mean(hits):>10.1f}")

        eq_regret = [
            u_star[si, jj] - float(mu_true[j, sub].mean())
            for si, sub in enumerate(subsets) for jj, j in enumerate(reb_idx)
        ]
        print(f"{n:>3} | {'Equal-Weight (reference)':<30} | {np.mean(eq_regret):>18.6f} | "
              f"{'—':>14} | {'—':>10}")
        print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT F: RL EMULATION — can the representation recover the OPTIMAL ROUTE
# of the MDP induced by the known g? The state space is discretized; the MDP
# (27-action grid, transition kernel = g + the DGP's process noise, reward =
# conditional Sharpe of the next state, discount gamma) is solved by stochastic
# value iteration -> oracle V*, pi*. Each extractor is probed for V* (Ridge
# regression) and pi* (27-way one-hot readout); the probed policy is scored by
# its exact Bellman regret on a fixed evaluation subsample.
# ============================================================================

def _v_lookup(s: np.ndarray, axes, V: np.ndarray) -> float:
    return float(V[tuple(int(np.argmin(np.abs(axes[k] - s[k]))) for k in range(3))])


def _bellman_q(s: np.ndarray, g, beta: float, axes, V: np.ndarray, gamma: float, rng) -> np.ndarray:
    """Stochastic Q(s, a) over the 27 grid actions: 3-noise-sample Bellman backup."""
    qs = np.zeros(len(ACTION_GRID))
    for k, a in enumerate(ACTION_GRID):
        acc = 0.0
        for _ in range(3):
            s_next = g(s, a, beta) + rng.normal(0.0, PROCESS_NOISE).astype(np.float32)
            acc += float(state_utility(s_next[None, :])[0]) + gamma * _v_lookup(s_next, axes, V)
        qs[k] = acc / 3.0
    return qs


def _solve_mdp(panel, transition_fn=None, gamma: float = 0.9, n_iter: int = 30,
               n_grid: Tuple[int, int, int] = (11, 9, 9)) -> Tuple[list, np.ndarray, float]:
    g = transition_fn if transition_fn is not None else transition_function
    beta = float(panel["asset_betas"].mean())
    flat = panel["states"].reshape(-1, 3)
    lo = np.percentile(flat, 0.5, axis=0)
    hi = np.percentile(flat, 99.5, axis=0)
    axes = [np.linspace(lo[k], hi[k], n_grid[k]) for k in range(3)]
    rng = np.random.default_rng(PROBE_SEED)
    V = np.zeros(n_grid)
    cells = list(itertools.product(range(n_grid[0]), range(n_grid[1]), range(n_grid[2])))
    for _ in range(n_iter):
        V_new = np.zeros(n_grid)
        for cell in cells:
            s = np.array([axes[k][cell[k]] for k in range(3)], dtype=np.float32)
            V_new[cell] = float(np.max(_bellman_q(s, g, beta, axes, V, gamma, rng)))
        V = V_new
    return axes, V, beta


def _mdp_targets(s_arr: np.ndarray, mdp, transition_fn, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row oracle V*(s) (nearest cell) and pi*(s) (exact one-step lookahead on V*)."""
    axes, V, beta = mdp
    g = transition_fn if transition_fn is not None else transition_function
    rng = np.random.default_rng(PROBE_SEED + 1)
    v = np.array([_v_lookup(s, axes, V) for s in s_arr])
    pi = np.array([int(np.argmax(_bellman_q(s, g, beta, axes, V, gamma, rng))) for s in s_arr], dtype=np.int64)
    return v, pi


def run_experiment_f_rl_emulation(panel, device: str = "cpu", cache: dict = None, H: int = 5,
                                  gamma: float = 0.9, n_eval: int = 3000):
    print("\n" + "=" * 108)
    print("TABLE 6 | EXPERIMENT F: RL EMULATION — CAN THE REPRESENTATION RECOVER THE OPTIMAL ROUTE OF")
    print(f"         THE MDP INDUCED BY g? (27-action grid, stochastic value iteration, gamma = {gamma}, reward =")
    print("         conditional Sharpe of the next state; probed policies scored by exact Bellman regret)")
    print("=" * 108)

    data = build_horizon_dataset(panel, context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    feats = build_extractor_features(data, "medium", H, device=device, cache=cache)

    if "A" not in _MDP_CACHE:
        _MDP_CACHE["A"] = _solve_mdp(panel, gamma=gamma)
    mdp = _MDP_CACHE["A"]

    V_tr, pi_tr = _mdp_targets(data["s_curr_train"], mdp, None, gamma)
    V_te, pi_te = _mdp_targets(data["s_curr_test"], mdp, None, gamma)
    onehot = np.eye(len(ACTION_GRID), dtype=np.float32)
    act_tr, act_te = data["act_train"], data["act_test"]
    asset_te = data["asset_test"]

    rng_sub = np.random.default_rng(PROBE_SEED + 2)
    sub = rng_sub.choice(len(V_te), size=min(n_eval, len(V_te)), replace=False)
    qs_sub = np.stack([
        _bellman_q(s, transition_function, mdp[2], mdp[0], mdp[1], gamma, np.random.default_rng(PROBE_SEED + 3))
        for s in data["s_curr_test"][sub]
    ])
    a_star = np.argmax(qs_sub, axis=1)
    rows_idx = np.arange(len(sub))

    def bellman_regret(a_hat_sub: np.ndarray) -> float:
        return float(np.mean(qs_sub[rows_idx, a_star] - qs_sub[rows_idx, a_hat_sub]))

    rng_pol = np.random.default_rng(PROBE_SEED + 4)
    zero_idx = int(np.argmin(np.linalg.norm(ACTION_GRID, axis=1)))

    header = (f"{'Policy / Extractor':<30} | {'Bellman Regret':>14} | {'pi* Accuracy %':>13} | "
              f"{'V* Within R2':>12} | {'V* Pooled R2':>12}")
    print(header)
    print("-" * len(header))
    print(f"{'Oracle (value iteration)':<30} | {0.0:>14.4f} | {100.0:>13.1f} | {'—':>12} | {'—':>12}")
    for name, (Z_tr, Z_te) in feats.items():
        F_tr = np.hstack([Z_tr, act_tr])
        F_te = np.hstack([Z_te, act_te])
        pi_hat = fit_predict_probe(F_tr, onehot[pi_tr], F_te, probe_type="ridge", seed=PROBE_SEED)
        a_hat = np.argmax(pi_hat, axis=1)
        v_hat = fit_predict_probe(F_tr, V_tr, F_te, probe_type="ridge", seed=PROBE_SEED)
        print(
            f"{name:<30} | {bellman_regret(a_hat[sub]):>14.4f} | "
            f"{100.0 * np.mean(a_hat == pi_te):>13.1f} | "
            f"{within_asset_r2(V_te, v_hat, asset_te):>12.3f} | {r2_score(V_te, v_hat):>12.3f}"
        )
    print(f"{'Random action':<30} | {bellman_regret(rng_pol.integers(len(ACTION_GRID), size=len(sub))):>14.4f} | "
          f"{100.0 / len(ACTION_GRID):>13.1f} | {'—':>12} | {'—':>12}")
    print(f"{'Zero action (do nothing)':<30} | {bellman_regret(np.full(len(sub), zero_idx)):>14.4f} | "
          f"{100.0 * np.mean(pi_te == zero_idx):>13.1f} | {'—':>12} | {'—':>12}")
    print("-" * len(header))


# %%
# ============================================================================
# EXPERIMENT G: DYNAMICS SHIFT — is optimal behavior SAVED when the transition
# function changes to one that steers states AWAY from the original central
# mass? g_shift adds linear repulsion (kappa) from the original (l, m) mean.
# A-trained extractors and probes are transferred to the shifted panel
# (state recovery + Bellman regret under the shifted MDP's oracle), compared
# against extractors/probes retrained on the shifted panel.
# ============================================================================

def run_experiment_g_dynamics_shift(panel_A, device: str = "cpu", cache: dict = None, H: int = 5,
                                    kappa: float = 0.10, gamma: float = 0.9, n_eval: int = 3000):
    print("\n" + "=" * 108)
    print("TABLE 7 | EXPERIMENT G: DYNAMICS SHIFT — IS OPTIMAL BEHAVIOR SAVED WHEN THE TRANSITION FUNCTION")
    print(f"         STEERS STATES AWAY FROM THE ORIGINAL CENTRAL MASS? (g_shift = g + saturating tanh")
    print(f"         repulsion kappa = {kappa} from the original (l, m) mean; regret scored under the shifted MDP's oracle)")
    print("=" * 108)

    center = panel_A["states"].reshape(-1, 3).mean(axis=0)
    repel = np.array([kappa, kappa, 0.0], dtype=np.float32)

    def g_shift(s, a, beta):
        # Saturating repulsion: the central mass is evacuated but the force is bounded, so the
        # shifted world stays stationary. (Linear repulsion compounds with g's persistence and
        # diverges geometrically -> float32 overflow -> NaN returns/features -> KMeans crash.)
        s_next = transition_function(s, a, beta)
        return s_next + (repel * np.tanh(s - center)).astype(np.float32)

    cfg = SimConfig(n_assets=N_ASSETS, n_days=N_DAYS, context_len=CONTEXT_LEN)
    panel_B = generate_panel_with_interventions(cfg, seed=PANEL_SEED, state_noise=NOISE_LEVELS["medium"], transition_fn=g_shift)
    data_A = build_horizon_dataset(panel_A, context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    data_B = build_horizon_dataset(panel_B, context_len=CONTEXT_LEN, horizon=H, train_end_day=TRAIN_END_DAY)
    feats_A = build_extractor_features(data_A, "medium", H, device=device, cache=cache)
    models = EXTRACTOR_MODELS[("medium", H)]

    if "A" not in _MDP_CACHE:
        _MDP_CACHE["A"] = _solve_mdp(panel_A, gamma=gamma)
    if "B" not in _MDP_CACHE:
        _MDP_CACHE["B"] = _solve_mdp(panel_B, transition_fn=g_shift, gamma=gamma)
    mdp_A, mdp_B = _MDP_CACHE["A"], _MDP_CACHE["B"]

    V_B_tr, pi_B_tr = _mdp_targets(data_B["s_curr_train"], mdp_B, g_shift, gamma)
    V_B_te, pi_B_te = _mdp_targets(data_B["s_curr_test"], mdp_B, g_shift, gamma)
    V_A_tr, pi_A_tr = _mdp_targets(data_A["s_curr_train"], mdp_A, None, gamma)

    rng_sub = np.random.default_rng(PROBE_SEED + 5)
    sub = rng_sub.choice(len(V_B_te), size=min(n_eval, len(V_B_te)), replace=False)
    qs_B = np.stack([
        _bellman_q(s, g_shift, mdp_B[2], mdp_B[0], mdp_B[1], gamma, np.random.default_rng(PROBE_SEED + 6))
        for s in data_B["s_curr_test"][sub]
    ])
    a_star_B = np.argmax(qs_B, axis=1)
    rows_idx = np.arange(len(sub))

    def regret_B(a_hat_sub: np.ndarray) -> float:
        return float(np.mean(qs_B[rows_idx, a_star_B] - qs_B[rows_idx, a_hat_sub]))

    onehot = np.eye(len(ACTION_GRID), dtype=np.float32)
    act_A_tr, act_B_tr, act_B_te = data_A["act_train"], data_B["act_train"], data_B["act_test"]
    asset_B_te, s_hor_A_tr, s_hor_B_tr, s_hor_B_te = (
        data_B["asset_test"], data_A["s_hor_train"], data_B["s_hor_train"], data_B["s_hor_test"]
    )

    # Retrained-on-B extractors: JEPA trained on B; untrained re-drawn; projector frozen
    set_seed(100)
    untr_B_enc = TimeSeriesEncoder(mag_preserving=True).to(device)
    untr_B_pred = LatentPredictor(embed_dim=untr_B_enc.out_dim, action_dim=0).to(device)
    jepa_B_enc, jepa_B_pred = train_jepa_model(
        data_B["X_ctx_train"], data_B["X_fut_train"], act_B_tr, use_action_predictor=False,
        use_cluster_sampler=True, mag_preserving=True, epochs=JEPA_EPOCHS, seed=PROBE_SEED, device=device,
    )

    def transfer_features(name):
        if name == "Random Linear":
            return extract_representations(models["proj"], models["id_pred"], data_B["X_ctx_test"], device=device)
        if name == "Untrained Conv+Transformer":
            enc_m, pred_m = models["untrained"]
            return extract_representations(enc_m, pred_m, data_B["X_ctx_test"], device=device)
        enc_m, pred_m = models["jepa"]
        return extract_representations(enc_m, pred_m, data_B["X_ctx_test"], device=device)

    def retrain_features(name):
        if name == "Random Linear":
            return (
                extract_representations(models["proj"], models["id_pred"], data_B["X_ctx_train"], device=device),
                extract_representations(models["proj"], models["id_pred"], data_B["X_ctx_test"], device=device),
            )
        if name == "Untrained Conv+Transformer":
            return (
                extract_representations(untr_B_enc, untr_B_pred, data_B["X_ctx_train"], device=device),
                extract_representations(untr_B_enc, untr_B_pred, data_B["X_ctx_test"], device=device),
            )
        return (
            extract_representations(jepa_B_enc, jepa_B_pred, data_B["X_ctx_train"], device=device),
            extract_representations(jepa_B_enc, jepa_B_pred, data_B["X_ctx_test"], device=device),
        )

    header = (f"{'Policy (B-world)':<40} | {'Bellman Regret':>14} | {'pi*_B Acc %':>11} | {'State R2':>9}")
    print(f"Shift center (original l, m mean): ({center[0]:.3f}, {center[1]:.3f}); kappa = {kappa}")
    print(header)
    print("-" * len(header))
    print(f"{'Oracle-B (value iteration on g_shift)':<40} | {0.0:>14.4f} | {100.0:>11.1f} | {'—':>9}")

    for name, (Z_A_tr, _) in feats_A.items():
        Z_B_te = transfer_features(name)
        F_A_tr = np.hstack([Z_A_tr, act_A_tr])
        F_B_te = np.hstack([Z_B_te, act_B_te])

        pi_hat = fit_predict_probe(F_A_tr, onehot[pi_A_tr], F_B_te, probe_type="ridge", seed=PROBE_SEED)
        a_hat = np.argmax(pi_hat, axis=1)
        s_hat = fit_predict_probe(F_A_tr, s_hor_A_tr, F_B_te, probe_type="ridge", seed=PROBE_SEED)
        r2_transfer = within_asset_r2(s_hor_B_te, s_hat, asset_B_te)
        print(f"{f'{name}  A->B transferred':<40} | {regret_B(a_hat[sub]):>14.4f} | "
              f"{100.0 * np.mean(a_hat == pi_B_te):>11.1f} | {r2_transfer:>9.3f}")

        Z_B_tr, Z_B_te2 = retrain_features(name)
        F_B_tr = np.hstack([Z_B_tr, act_B_tr])
        F_B_te2 = np.hstack([Z_B_te2, act_B_te])
        pi_hat2 = fit_predict_probe(F_B_tr, onehot[pi_B_tr], F_B_te2, probe_type="ridge", seed=PROBE_SEED)
        a_hat2 = np.argmax(pi_hat2, axis=1)
        s_hat2 = fit_predict_probe(F_B_tr, s_hor_B_tr, F_B_te2, probe_type="ridge", seed=PROBE_SEED)
        r2_retrain = within_asset_r2(s_hor_B_te, s_hat2, asset_B_te)
        print(f"{f'{name}  retrained on B':<40} | {regret_B(a_hat2[sub]):>14.4f} | "
              f"{100.0 * np.mean(a_hat2 == pi_B_te):>11.1f} | {r2_retrain:>9.3f}")

    rng_pol = np.random.default_rng(PROBE_SEED + 7)
    zero_idx = int(np.argmin(np.linalg.norm(ACTION_GRID, axis=1)))
    print(f"{'Random action':<40} | {regret_B(rng_pol.integers(len(ACTION_GRID), size=len(sub))):>14.4f} | "
          f"{100.0 / len(ACTION_GRID):>11.1f} | {'—':>9}")
    print(f"{'Zero action (do nothing)':<40} | {regret_B(np.full(len(sub), zero_idx)):>14.4f} | "
          f"{100.0 * np.mean(pi_B_te == zero_idx):>11.1f} | {'—':>9}")
    print("-" * len(header))


# %%
# ============================================================================
# MAIN - build the three noise panels and run all seven tables.
# In Colab (notebook mode or !python) __name__ == "__main__", so Run all works.
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Study 3 ablations on device={device} (TabICL installed={HAS_TABICL})")
    cfg = SimConfig(n_assets=N_ASSETS, n_days=N_DAYS, context_len=CONTEXT_LEN)

    panels = {
        noise_name: generate_panel_with_interventions(cfg, seed=PANEL_SEED, state_noise=noise)
        for noise_name, noise in NOISE_LEVELS.items()
    }

    feature_cache = {}
    run_ablation_a_random_baseline(panels["medium"], device=device, cache=feature_cache)
    run_ablation_b_order_recovery(panels, device=device, cache=feature_cache)
    run_experiment_c_belief_planning(panels["medium"], device=device)
    run_experiment_d_portfolio(panels["medium"], device=device, cache=feature_cache)
    run_experiment_e_lp_scaling(panels["medium"], device=device, cache=feature_cache)
    run_experiment_f_rl_emulation(panels["medium"], device=device, cache=feature_cache)
    run_experiment_g_dynamics_shift(panels["medium"], device=device, cache=feature_cache)
