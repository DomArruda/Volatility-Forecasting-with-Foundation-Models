import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# Attempt to import TabICL; provide a faithful In-Context Kernel/Ensemble fallback if absent
try:
    from tabicl import TabICLRegressor
    HAS_TABICL = True
except ImportError:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline
    HAS_TABICL = False


# ============================================================================
# 1. UTILITIES: SEEDS, EFFECTIVE RANK (Roy & Vetterli, 2007), DIEBOLD-MARIANO
# ============================================================================

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


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 5) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Newey-West HAC variance (lag = max(1, 2*h)).
    Returns (mean_diff A - B, t_stat). Negative favours A.
    """
    d = loss_a - loss_b
    T = len(d)
    d_bar = float(np.mean(d))
    d_centered = d - d_bar
    max_lag = min(T - 1, max(1, 2 * h))
    gamma_0 = np.dot(d_centered, d_centered) / T
    nw_var = gamma_0
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)
        gamma_k = np.dot(d_centered[k:], d_centered[:-k]) / T
        nw_var += 2.0 * weight * gamma_k
    nw_var = max(nw_var / T, 1e-12)
    t_stat = d_bar / math.sqrt(nw_var)
    return d_bar, t_stat


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Mean and sample std (ddof=1) across seeds; std=0.0 for a single seed."""
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


# ============================================================================
# 2. SYNTHETIC STATE-FUNCTION & MULTI-TIMESCALE ENVIRONMENT
# ============================================================================

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

            s_det = transition_function(s, a_t, asset_betas[i])
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


# ============================================================================
# 3. JEPA ENCODER, ACTION-CONDITIONED PREDICTOR & CLUSTER BATCH SAMPLER
# ============================================================================

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


# ============================================================================
# 4. PROBES: TABICL (WITH SUPPORT STRIDE) & RIDGE + WITHIN-ASSET R2
# ============================================================================

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


def fit_probe_once_predict_many(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test_list: List[np.ndarray],
    probe_type: str = "tabicl",
    support_stride: int = 3,
    max_support: int = 3000,
    seed: int = 0,
) -> List[np.ndarray]:
    scaler = StandardScaler()
    Z_tr_s = scaler.fit_transform(Z_train)
    Z_te_s_list = [scaler.transform(Z_te) for Z_te in Z_test_list]

    if probe_type == "ridge":
        model = Ridge(alpha=10.0, random_state=seed)
        model.fit(Z_tr_s, y_train)
        return [model.predict(Z_te_s) for Z_te_s in Z_te_s_list]

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
            return [reg.predict(Z_te_s) for Z_te_s in Z_te_s_list]
        else:
            models = []
            for d in range(y_sup.shape[1]):
                reg = TabICLRegressor(n_estimators=4, random_state=seed)
                reg.fit(X_sup, y_sup[:, d])
                models.append(reg)
            out_list = []
            for Z_te_s in Z_te_s_list:
                preds = [models[d].predict(Z_te_s) for d in range(y_sup.shape[1])]
                out_list.append(np.column_stack(preds))
            return out_list
    else:
        pipes = []
        for est in range(4):
            pipe = make_pipeline(
                RBFSampler(gamma=1.0 / Z_tr_s.shape[1], n_components=256, random_state=seed + est),
                Ridge(alpha=5.0)
            )
            pipe.fit(X_sup, y_sup)
            pipes.append(pipe)
        return [np.mean([p.predict(Z_te_s) for p in pipes], axis=0) for Z_te_s in Z_te_s_list]


def within_asset_r2(y_true: np.ndarray, y_pred: np.ndarray, asset_ids: np.ndarray) -> float:
    yt_dem = y_true.copy()
    yp_dem = y_pred.copy()
    for aid in np.unique(asset_ids):
        mask = asset_ids == aid
        yt_dem[mask] -= np.mean(yt_dem[mask], axis=0)
        yp_dem[mask] -= np.mean(yp_dem[mask], axis=0)
    return float(r2_score(yt_dem, yp_dem))


# ============================================================================
# 5. EXPERIMENT 1: FUNCTION / STATE-TRANSITION REPLICATION
# ============================================================================

def run_experiment_1_state_transition(
    panel: Dict[str, np.ndarray],
    device: str = "cpu",
    seeds: Tuple[int, ...] = (0, 1, 2),
):
    print("=" * 100)
    print("EXPERIMENT 1: CAN JEPA + TabICL REPLICATE DAILY STATE-TRANSITION FUNCTIONS g(s_t, a_t)?")
    print(f"Protocol: {len(seeds)} seeds {tuple(seeds)}; mean ± sample std across seeds.")
    print("Panel (seed 2026) and counterfactual actions (rng 123) fixed; seeds vary encoder init,")
    print("JEPA training, and probe fits. Seed 0 reproduces the original single-run protocol.")
    print("=" * 100)
    data = build_horizon_dataset(panel, context_len=40, horizon=5, train_end_day=1000)

    X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
    X_ctx_te, X_fut_te, act_te = data["X_ctx_test"], data["X_fut_test"], data["act_test"]
    s_det1_tr, s_det1_te = data["s_det1_train"], data["s_det1_test"]
    s_curr_te = data["s_curr_test"]
    asset_te = data["asset_test"]
    betas = panel["asset_betas"]

    rng = np.random.default_rng(123)
    a_cf = rng.normal(0.0, 0.8, size=act_te.shape).astype(np.float32)
    a_zero = np.zeros_like(act_te)

    true_delta_s = np.array([
        transition_function(s_curr_te[k], a_cf[k], betas[asset_te[k]]) -
        transition_function(s_curr_te[k], a_zero[k], betas[asset_te[k]])
        for k in range(len(s_curr_te))
    ])

    metrics: Dict[str, Dict[str, List[float]]] = {}

    for seed in seeds:
        set_seed(seed)
        untrained_enc = TimeSeriesEncoder(mag_preserving=True).to(device)
        untrained_pred = LatentPredictor(embed_dim=untrained_enc.out_dim, action_dim=0).to(device)
        Z_untrained_tr = extract_representations(untrained_enc, untrained_pred, X_ctx_tr, device=device)
        Z_untrained_te = extract_representations(untrained_enc, untrained_pred, X_ctx_te, device=device)

        jepa_std_enc, jepa_std_pred = train_jepa_model(
            X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=False,
            use_cluster_sampler=True, mag_preserving=True, epochs=10, seed=seed, device=device
        )
        Z_jepa_std_tr = extract_representations(jepa_std_enc, jepa_std_pred, X_ctx_tr, device=device)
        Z_jepa_std_te = extract_representations(jepa_std_enc, jepa_std_pred, X_ctx_te, device=device)

        jepa_act_enc, jepa_act_pred = train_jepa_model(
            X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=True,
            use_cluster_sampler=True, mag_preserving=True, epochs=10, seed=seed, device=device
        )
        Z_jepa_act_tr = extract_representations(jepa_act_enc, jepa_act_pred, X_ctx_tr, act_tr, use_action=True, device=device)
        Z_jepa_act_te = extract_representations(jepa_act_enc, jepa_act_pred, X_ctx_te, act_te, use_action=True, device=device)
        Z_jepa_act_cf = extract_representations(jepa_act_enc, jepa_act_pred, X_ctx_te, a_cf, use_action=True, device=device)
        Z_jepa_act_zero = extract_representations(jepa_act_enc, jepa_act_pred, X_ctx_te, a_zero, use_action=True, device=device)

        models_exp1 = [
            (
                "A. Untrained Encoder [z_t, a_t] + Ridge", "ridge",
                np.hstack([Z_untrained_tr, act_tr]),
                [np.hstack([Z_untrained_te, act_te]), np.hstack([Z_untrained_te, a_cf]), np.hstack([Z_untrained_te, a_zero])]
            ),
            (
                "A. Untrained Encoder [z_t, a_t] + TabICL", "tabicl",
                np.hstack([Z_untrained_tr, act_tr]),
                [np.hstack([Z_untrained_te, act_te]), np.hstack([Z_untrained_te, a_cf]), np.hstack([Z_untrained_te, a_zero])]
            ),
            (
                "D. Standard JEPA [z_t, a_t] + TabICL", "tabicl",
                np.hstack([Z_jepa_std_tr, act_tr]),
                [np.hstack([Z_jepa_std_te, act_te]), np.hstack([Z_jepa_std_te, a_cf]), np.hstack([Z_jepa_std_te, a_zero])]
            ),
            (
                "E. Action-JEPA P(z_t, a_t) + Ridge", "ridge",
                Z_jepa_act_tr,
                [Z_jepa_act_te, Z_jepa_act_cf, Z_jepa_act_zero]
            ),
            (
                "E. Action-JEPA [P(z_t, a_t), a_t] + TabICL", "tabicl",
                np.hstack([Z_jepa_act_tr, act_tr]),
                [np.hstack([Z_jepa_act_te, act_te]), np.hstack([Z_jepa_act_cf, a_cf]), np.hstack([Z_jepa_act_zero, a_zero])]
            ),
        ]

        for name, p_type, F_tr, F_te_list in models_exp1:
            pred_obs, _, _ = fit_probe_once_predict_many(
                F_tr, s_det1_tr, F_te_list, probe_type=p_type, support_stride=5, seed=seed
            )
            if name not in metrics:
                metrics[name] = {"pool": [], "within": [], "mse": []}
            metrics[name]["pool"].append(r2_score(s_det1_te, pred_obs))
            metrics[name]["within"].append(within_asset_r2(s_det1_te, pred_obs, asset_te))
            metrics[name]["mse"].append(mean_squared_error(s_det1_te, pred_obs))

    header = f"{'Model Variant':<44} | {'Pooled State R²':>16} | {'Within-Asset R²':>16} | {'Transition MSE':>16}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        m_pool, s_pool = mean_std(m["pool"])
        m_with, s_with = mean_std(m["within"])
        m_mse, s_mse = mean_std(m["mse"])
        print(
            f"{name:<44} | {m_pool:>9.4f}±{s_pool:.4f} | {m_with:>9.4f}±{s_with:.4f} | {m_mse:>9.4f}±{s_mse:.4f}"
        )


# ============================================================================
# 6. EXPERIMENT 2: TIMESCALE SWEEP (H in {1, 5, 22, 60} DAYS)
# ============================================================================

def run_experiment_2_timescale_sweep(
    panel: Dict[str, np.ndarray],
    horizons: List[int] = [1, 5, 22, 60],
    device: str = "cpu",
    seeds: Tuple[int, ...] = (42, 43, 44),
):
    print("\n" + "=" * 118)
    print("EXPERIMENT 2: HOW REPRESENTATION RANK & ACCURACY CHANGE ACROSS TIMESCALES (H = 1, 5, 22, 60)")
    print(f"Protocol: {len(seeds)} seeds {tuple(seeds)}; mean ± sample std across seeds.")
    print("Panel (seed 2026) fixed; seeds vary encoder init, JEPA training, and probe fits. Each seed's")
    print("DM test uses that seed's untrained encoder as baseline. Seed 42 reproduces the original protocol.")
    print("=" * 118)

    for H in horizons:
        data = build_horizon_dataset(panel, context_len=40, horizon=H, train_end_day=1000)
        X_ctx_tr, X_fut_tr, act_tr = data["X_ctx_train"], data["X_fut_train"], data["act_train"]
        X_ctx_te, X_fut_te, act_te = data["X_ctx_test"], data["X_fut_test"], data["act_test"]
        y_rv_tr, y_rv_te = data["y_rv_train"], data["y_rv_test"]
        s_hor_tr, s_hor_te = data["s_hor_train"], data["s_hor_test"]
        asset_te, date_te = data["asset_test"], data["date_test"]
        unique_dates = np.unique(date_te)

        metrics: Dict[str, Dict[str, List[float]]] = {}

        for seed in seeds:
            set_seed(seed)
            enc_A = TimeSeriesEncoder(mag_preserving=True).to(device)
            pred_A = LatentPredictor(embed_dim=enc_A.out_dim, action_dim=0).to(device)
            Z_A_tr = extract_representations(enc_A, pred_A, X_ctx_tr, device=device)
            Z_A_te = extract_representations(enc_A, pred_A, X_ctx_te, device=device)

            enc_C, pred_C = train_jepa_model(
                X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=False,
                use_cluster_sampler=False, mag_preserving=True, epochs=10, seed=seed, device=device
            )
            Z_C_tr = extract_representations(enc_C, pred_C, X_ctx_tr, device=device)
            Z_C_te = extract_representations(enc_C, pred_C, X_ctx_te, device=device)

            enc_D, pred_D = train_jepa_model(
                X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=False,
                use_cluster_sampler=True, mag_preserving=True, epochs=10, seed=seed, device=device
            )
            Z_D_tr = extract_representations(enc_D, pred_D, X_ctx_tr, device=device)
            Z_D_te = extract_representations(enc_D, pred_D, X_ctx_te, device=device)

            enc_E, pred_E = train_jepa_model(
                X_ctx_tr, X_fut_tr, act_tr, use_action_predictor=True,
                use_cluster_sampler=True, mag_preserving=True, epochs=10, seed=seed, device=device
            )
            Z_E_tr = extract_representations(enc_E, pred_E, X_ctx_tr, act_tr, use_action=True, device=device)
            Z_E_te = extract_representations(enc_E, pred_E, X_ctx_te, act_te, use_action=True, device=device)

            variants = [
                ("A. Untrained + TabICL", Z_A_tr, Z_A_te),
                ("C. JEPA Uniform + TabICL", Z_C_tr, Z_C_te),
                ("D. JEPA Cluster + TabICL", Z_D_tr, Z_D_te),
                ("E. Action-JEPA + TabICL", Z_E_tr, Z_E_te),
            ]

            loss_by_date_A = None
            for v_name, Z_tr, Z_te in variants:
                erank = compute_effective_rank(Z_te)
                y_hat = fit_predict_probe(
                    np.hstack([Z_tr, act_tr]), y_rv_tr,
                    np.hstack([Z_te, act_te]), probe_type="tabicl",
                    support_stride=max(1, H // 2), seed=seed
                )
                r2_pool = r2_score(y_rv_te, y_hat)
                r2_with = within_asset_r2(y_rv_te, y_hat, asset_te)

                s_hat = fit_predict_probe(
                    np.hstack([Z_tr, act_tr]), s_hor_tr,
                    np.hstack([Z_te, act_te]), probe_type="tabicl",
                    support_stride=max(1, H // 2), seed=seed
                )
                r2_state = within_asset_r2(s_hor_te, s_hat, asset_te)

                sq_err = (y_rv_te - y_hat) ** 2
                date_losses = np.array([np.mean(sq_err[date_te == d]) for d in unique_dates])

                if v_name not in metrics:
                    metrics[v_name] = {"erank": [], "pool": [], "within": [], "state": [], "dm_t": []}
                metrics[v_name]["erank"].append(erank)
                metrics[v_name]["pool"].append(r2_pool)
                metrics[v_name]["within"].append(r2_with)
                metrics[v_name]["state"].append(r2_state)

                if loss_by_date_A is None:
                    loss_by_date_A = date_losses
                else:
                    _, t_stat = diebold_mariano_test(date_losses, loss_by_date_A, h=H)
                    metrics[v_name]["dm_t"].append(t_stat)

        header = (
            f"{'Horizon H':<10} | {'Variant':<28} | {'Eff. Rank':>13} | "
            f"{'RV Pooled R²':>15} | {'RV Within R²':>15} | {'State H R²':>15} | {'DM t vs Untrained':>17}"
        )
        print(header)
        print("-" * len(header))
        for v_name, m in metrics.items():
            m_er, s_er = mean_std(m["erank"])
            m_pool, s_pool = mean_std(m["pool"])
            m_with, s_with = mean_std(m["within"])
            m_state, s_state = mean_std(m["state"])
            if m["dm_t"]:
                m_t, s_t = mean_std(m["dm_t"])
                dm_str = f"{m_t:+.2f}±{s_t:.2f}"
            else:
                dm_str = "baseline"
            print(
                f"H = {H:<6d} | {v_name:<28} | {m_er:>7.2f}±{s_er:.2f} | "
                f"{m_pool:>8.4f}±{s_pool:.4f} | {m_with:>8.4f}±{s_with:.4f} | "
                f"{m_state:>8.4f}±{s_state:.4f} | {dm_str:>17}"
            )
        print("-" * len(header))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running JEPA + TabICL State-Function & Timescale Experiments on device={device} (TabICL installed={HAS_TABICL})")
    cfg = SimConfig(n_assets=20, n_days=1400, context_len=40)
    panel_data = generate_panel_with_interventions(cfg, seed=2026)

    run_experiment_1_state_transition(panel_data, device=device)
    run_experiment_2_timescale_sweep(panel_data, horizons=[1, 5, 22, 60], device=device)
