import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from efficient_kan.efficientkan import KANFET
from torchdiffeq import odeint
from Time_MMD.numerical import *
from kan_diffusion.kan import KAN
from dataclasses import dataclass


# neural_ode_energy_forecast.py
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standardize_fit(x: np.ndarray, eps: float = 1e-8):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + eps
    return mu, sd

def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd

# -----------------------------
# RK4 integrator (fixed-step)
# -----------------------------
@torch.no_grad()
def _check_finite(x, name="tensor"):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {name}")

def odeint_rk4(f, z0, t, n_substeps: int = 4):
    """
    Fixed-step RK4 integration from t[0] ... t[T-1].
    - f: (t_scalar, z) -> dz/dt, same shape as z
    - z0: (B, latent_dim)
    - t:  (T,) monotonic increasing
    Returns:
      z_traj: (T, B, latent_dim)
    """
    assert t.ndim == 1
    T = t.shape[0]
    B, D = z0.shape
    z = z0
    out = [z0]

    for i in range(T - 1):
        t0 = t[i]
        t1 = t[i + 1]
        h = (t1 - t0) / float(n_substeps)

        # sub-steps
        ti = t0
        for _ in range(n_substeps):
            k1 = f(ti, z)
            k2 = f(ti + 0.5 * h, z + 0.5 * h * k1)
            k3 = f(ti + 0.5 * h, z + 0.5 * h * k2)
            k4 = f(ti + h, z + h * k3)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ti = ti + h

        out.append(z)

    return torch.stack(out, dim=0)  # (T,B,D)

# -----------------------------
# Dataset: sliding windows
# -----------------------------
class EnergyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, context_len: int, pred_len: int):
        """
        X: (N, F) standardized numeric features (can include target column too)
        y: (N,) standardized target series
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.N = len(X)
        self.max_start = self.N - (context_len + pred_len) + 1
        if self.max_start <= 0:
            raise ValueError("Not enough rows for given context_len + pred_len.")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        s = idx
        c = self.context_len
        p = self.pred_len
        x_ctx = self.X[s : s + c]                      # (c, F)
        y_fut = self.y[s + c : s + c + p]              # (p,)
        return torch.from_numpy(x_ctx), torch.from_numpy(y_fut)

# -----------------------------
# Model
# -----------------------------
class ODEDynamics(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, t, z):
        # t: scalar tensor or float; z: (B, D)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        t_in = t.expand(z.shape[0], 1)  # (B,1)
        return self.net(torch.cat([z, t_in], dim=-1))

class LatentNeuralODEForecaster(nn.Module):
    def __init__(self, num_features: int, context_len: int, pred_len: int,
                 latent_dim: int = 64, enc_hidden: int = 128, dec_hidden: int = 128,
                 dyn_hidden: int = 128):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        # Encode past window -> z0
        self.encoder = nn.Sequential(
            nn.Flatten(),  # (B, context_len*num_features)
            nn.Linear(context_len * num_features, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, latent_dim),
        )

        # ODE dynamics in latent space
        self.dynamics = ODEDynamics(latent_dim=latent_dim, hidden=dyn_hidden)

        # Decode latent trajectory -> target value at each future step
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, 1),
        )

    def forward(self, x_ctx, t_fut, rk4_substeps: int = 4):
        """
        x_ctx: (B, context_len, F)
        t_fut: (pred_len,) times to predict at (e.g., 0..pred_len-1)
        Returns:
          y_hat: (B, pred_len)
        """
        B = x_ctx.shape[0]
        z0 = self.encoder(x_ctx)  # (B, D)

        # Integrate from t=0 to future times
        z_traj =  odeint(self.dynamics, z0, t_fut, method="dopri5") #odeint_rk4(self.dynamics, z0, t_fut, n_substeps=rk4_substeps)  # (T,B,D)

        # decode each time step
        y_hat = self.decoder(z_traj)  # (T,B,1)
        y_hat = y_hat.squeeze(-1).transpose(0, 1)  # (B,T)
        return y_hat


@torch.no_grad()
def forecast_trend_plot(
    plot_name, 
    model,
    X,                 # standardized X (N,F)
    y_raw,             # original units (N,)
    context_len,
    pred_len,
    t_fut,             # torch tensor (pred_len,) on same device
    y_mu, y_sd,        # train-fitted scalers (shape (1,) each)
    train_end,
    val_end,
    test_end,
    device,
    rk4_substeps=4,
    horizon=7,         # 0 = first step of the pred_len forecast
    stride=1,          # increase to thin points
):
    model.eval()

    N = len(y_raw)
    pred_series = np.full(N, np.nan, dtype=np.float32)

    last_start = N - (context_len + pred_len)
    for s in range(0, last_start + 1, stride):
        x_ctx = torch.from_numpy(X[s : s + context_len]).unsqueeze(0).to(device).float()  # (1,c,F)
        y_hat_std = model(x_ctx, t_fut).squeeze(0)  # (pred_len,)
        y_hat = (y_hat_std[horizon].item() * float(y_sd[0])) + float(y_mu[0])

        # prediction corresponds to time index s + context_len + horizon
        ti = s + context_len + horizon
        pred_series[ti] = y_hat

    plt.figure()
    plt.plot(y_raw, label="ground truth")
    plt.plot(pred_series, label=f"forecast (horizon={horizon+1})")

    # separator lines
    plt.axvline(train_end, linestyle="--", label="train end")
    #plt.axvline(val_end, linestyle="--", label="val end")

    plt.title(plot_name)
    plt.xlabel("time index")
    plt.ylabel("Drought Factor")
    plt.legend()
    plt.savefig(f"Time_MMD_plots/Climate/horizon12/{plot_name}", bbox_inches="tight")
    plt.show()


# -----------------------------
# Training / Eval
# -----------------------------
@dataclass
class TrainConfig:
    model: nn.Module
    csv_path: str = "Time_MMD/numerical/Climate/Climate.csv"
    target_col: str = "OT"
    context_len: int = 32
    pred_len: int = 8
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    latent_dim: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    rk4_substeps: int = 4
    val_ratio: float = 0.15
    test_ratio: float = 0.15



def load_climate_csv(
    csv_path: str,
    target_col: str = "OT",
    area: str = "Total",     # set None to keep all areas
    drop_cols: tuple[str, ...] = ("MapDate", "StatisticFormatID"),  # avoid date/id leakage
):
    df = pd.read_csv(csv_path)

    # Climate.csv often has MapDate like 20240514 (int)
    if "MapDate" in df.columns:
        df["date"] = pd.to_datetime(df["MapDate"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Optional: parse these if you care about them
    for c in ["ValidStart", "ValidEnd", "start_date", "end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Optional: filter a single region to keep things simple
    if area is not None and "AreaOfInterest" in df.columns:
        df = df[df["AreaOfInterest"] == area].copy()

    # Keep numeric columns only
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Drop date-like / id-like numeric columns (optional but recommended)
    for c in drop_cols:
        if c in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[c])

    if target_col not in numeric_df.columns:
        raise ValueError(
            f"target_col={target_col!r} must be numeric and present. "
            f"Numeric columns are: {list(numeric_df.columns)}"
        )

    # Sort by time if we have it
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        # numeric_df must match df order (rebuild after sorting)
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        for c in drop_cols:
            if c in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[c])

    X = numeric_df.values
    y = numeric_df[target_col].values
    return df, numeric_df, X, y

def split_time_series(N: int, val_ratio: float, test_ratio: float):
    n_test = int(round(N * test_ratio))
    n_val  = int(round(N * val_ratio))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough data after split.")
    return n_train, n_train + n_val, N  # train_end, val_end, test_end

def train_and_predict(cfg: TrainConfig):
    set_seed(cfg.seed)

    df_raw, df_num, X_raw, y_raw = load_climate_csv(cfg.csv_path, cfg.target_col)
    N, F = X_raw.shape
    #print(f"Loaded Energy data: {N} rows, {F} numeric features.")
    train_end, val_end, test_end = split_time_series(N, cfg.val_ratio, cfg.test_ratio)

    # Fit scalers on TRAIN ONLY (time-series safe)
    X_mu, X_sd = standardize_fit(X_raw[:train_end])
    y_mu, y_sd = standardize_fit(y_raw[:train_end].reshape(-1, 1))
    y_mu = y_mu.reshape(-1)  # (1,)
    y_sd = y_sd.reshape(-1)

    X = standardize_apply(X_raw, X_mu, X_sd)
    y = standardize_apply(y_raw.reshape(-1, 1), y_mu.reshape(1,1), y_sd.reshape(1,1)).reshape(-1)

    # Build windowed datasets (note: each split uses its own contiguous range)
    def make_split(start, end):
        Xs = X[start:end]
        ys = y[start:end]
        return EnergyWindowDataset(Xs, ys, cfg.context_len, cfg.pred_len)

    # For val/test we start earlier so windows have enough context inside split
    # (Simpler: just slice split and let dataset enforce length; we keep contiguous.)
    train_ds = make_split(0, train_end)
    val_ds   = make_split(train_end - cfg.context_len - cfg.pred_len + 1, val_end)
    test_ds  = make_split(val_end - cfg.context_len - cfg.pred_len + 1, test_end)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = cfg.model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # future times: 0..pred_len-1 (assumes uniform sampling; Energy.csv looks weekly)
    t_fut = torch.linspace(0.0, float(cfg.pred_len - 1), steps=cfg.pred_len, device=cfg.device)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        n = 0
        for x_ctx, y_fut in loader:
            x_ctx = x_ctx.to(cfg.device)              # (B,c,F)
            y_fut = y_fut.to(cfg.device)              # (B,p)

            if train:
                opt.zero_grad(set_to_none=True)

            y_hat = model(x_ctx, t_fut, rk4_substeps=cfg.rk4_substeps)  # (B,p)
            loss = loss_fn(y_hat, y_fut)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item() * x_ctx.size(0)
            n += x_ctx.size(0)
        return total_loss / max(1, n)

    best_val = float("inf")
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        train_losses.append(tr)
        val_losses.append(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {tr:.5f} | val MSE {va:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    te = run_epoch(test_loader, train=False)
    print(f"Test MSE (standardized): {te:.5f}")

  
    end_idx = N - cfg.pred_len
    start_idx = end_idx - cfg.context_len

    model.eval()
    with torch.no_grad():
        x_last = torch.from_numpy(X[start_idx:end_idx]).unsqueeze(0).to(cfg.device).float()  # (1,c,F)
        y_hat_std = model(x_last, t_fut, rk4_substeps=cfg.rk4_substeps).squeeze(0).cpu().numpy()

    # de-standardize predictions back to original units
    y_hat = y_hat_std * y_sd[0] + y_mu[0]

    # perfectly aligned ground truth
    y_true = y_raw[end_idx : end_idx + cfg.pred_len]


    forecast_trend_plot(
        plot_name="MLP-Style Neural ODE Climate Forecast vs ground truths",
        model=model,
        X=X,                    # standardized X (NOT X_raw)
        y_raw=y_raw,            # original target units
        context_len=cfg.context_len,
        pred_len=cfg.pred_len,
        t_fut=t_fut,
        y_mu=y_mu,
        y_sd=y_sd,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        device=cfg.device,
        rk4_substeps=cfg.rk4_substeps,
        horizon=cfg.pred_len - 1,
        stride=1,
    )

    return {
        "model": model,
        "pred_future": y_hat,
        "true_future": y_true,
        "target_col": cfg.target_col,
        "context_len": cfg.context_len,
        "pred_len": cfg.pred_len,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": te,
    }




def plot_train_val_loss(out, diffusion_out, kan_diffusion_out, kan_fet_diffusion_out):
    """
    Compare ODE vs Diffusion training/validation curves.

    ODE:
      - train_losses: MSE
      - val_losses:   MSE

    Diffusion:
      - train_diff_loss: epsilon-prediction loss
      - val_mse_std:     sampling MSE (standardized)
    """

    ode_train = np.array(out["train_losses"])
    ode_val   = np.array(out["val_losses"])

    diff_train = np.array(diffusion_out["train_diff_loss"])
    diff_val   = np.array(diffusion_out["val_mse_std"])

    kan_train = np.array(kan_diffusion_out["train_diff_loss"])
    kan_val   = np.array(kan_diffusion_out["val_mse_std"])

    kan_fet_train = np.array(kan_fet_diffusion_out["train_diff_loss"])
    kan_fet_val   = np.array(kan_fet_diffusion_out["val_mse_std"])
    # Epoch axes (allow different lengths safely)
    e_ode  = np.arange(1, len(ode_train) + 1)
    e_diff = np.arange(1, len(diff_train) + 1)

    plt.figure(figsize=(8, 5))

    # ----- ODE curves -----
    plt.plot(e_ode, ode_train, label="ODE train (MSE)", color="tab:blue")
    plt.plot(e_ode, ode_val,   label="ODE val (MSE)",   color="tab:blue", linestyle="--")
    
    # ----- Diffusion curves -----
    plt.plot(e_diff, diff_train, label="Diffusion train (ε-loss)", color="tab:orange")
    plt.plot(e_diff, diff_val,   label="Diffusion val (MSE)",      color="tab:orange", linestyle="--")

    # ----- KAN Diffusion curves -----
    plt.plot(e_diff, kan_train, label="KAN Diffusion train (ε-loss)", color="tab:green")
    plt.plot(e_diff, kan_val,   label="KAN Diffusion val (MSE)",      color="tab:green", linestyle="--")

    plt.plot(e_diff, kan_fet_train, label="KAN-FET Diffusion train (ε-loss)", color="tab:red")
    plt.plot(e_diff, kan_fet_val,   label="KAN-FET Diffusion val (MSE)",      color="tab:red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (standardized)")
    plt.title("ODE vs Diffusion vs KAN Diffusion vs KAN-FET Diffusion: Train / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import math
import torch


# -----------------------------
# Utility: sinusoidal t embedding for diffusion step
# -----------------------------
def sinusoidal_emb(t: torch.Tensor, dim: int):
    """
    t: (B,) int/float tensor
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# -----------------------------
# Gaussian Diffusion (DDPM) over y-sequence
# -----------------------------
class GaussianDiffusion1D:
    """
    Diffusion over y sequence: y0 shape (B, pred_len)
    Condition: cond shape (B, C) (you provide it)
    """
    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register = {}
        self.register["betas"] = betas
        self.register["alphas"] = alphas
        self.register["alphas_bar"] = alphas_bar
        self.register["sqrt_alphas_bar"] = torch.sqrt(alphas_bar)
        self.register["sqrt_one_minus_alphas_bar"] = torch.sqrt(1.0 - alphas_bar)

        # For sampling
        self.register["sqrt_recip_alphas"] = torch.sqrt(1.0 / alphas)
        self.register["posterior_variance"] = betas * (1.0 - torch.cat([alphas_bar[:1], alphas_bar[:-1]])) / (1.0 - alphas_bar)

    def to(self, device):
        for k, v in self.register.items():
            self.register[k] = v.to(device)
        return self

    def q_sample(self, y0, t_idx, noise=None):
        """
        y_t = sqrt(a_bar)*y0 + sqrt(1-a_bar)*eps
        y0: (B,P)
        t_idx: (B,) integer diffusion step in [0, T-1]
        """
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_ab = self.register["sqrt_alphas_bar"][t_idx].unsqueeze(1)          # (B,1)
        sqrt_omab = self.register["sqrt_one_minus_alphas_bar"][t_idx].unsqueeze(1)
        return sqrt_ab * y0 + sqrt_omab * noise, noise

    @torch.no_grad()
    def p_sample(self, eps_model, y_t, t_idx, cond):
        """
        One reverse step: predict eps, compute mean, add noise (except at t=0)
        """
        betas_t = self.register["betas"][t_idx].unsqueeze(1)
        sqrt_recip_alpha_t = self.register["sqrt_recip_alphas"][t_idx].unsqueeze(1)
        alphas_bar_t = self.register["alphas_bar"][t_idx].unsqueeze(1)

        # predict eps
        eps_hat = eps_model(y_t, t_idx, cond)  # (B,P)

        # DDPM mean
        # mu = 1/sqrt(alpha_t) * (y_t - beta_t/sqrt(1-a_bar_t) * eps_hat)
        mu = sqrt_recip_alpha_t * (y_t - betas_t * eps_hat / torch.sqrt(1.0 - alphas_bar_t))

        if (t_idx == 0).all():
            return mu

        var = self.register["posterior_variance"][t_idx].unsqueeze(1).clamp_min(1e-20)
        noise = torch.randn_like(y_t)
        return mu + torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, eps_model, shape, cond, device):
        """
        Start from N(0,1) and sample down to y0.
        shape: (B,P)
        """
        y = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_idx = torch.full((shape[0],), t, device=device, dtype=torch.long)
            y = self.p_sample(eps_model, y, t_idx, cond)
        return y

# -----------------------------
# Diffusion epsilon-predictor head
# -----------------------------
class DiffusionEpsHead(nn.Module):
    """
    Predict eps in R^{pred_len}, conditioned on:
      - current noisy y_t (B,P)
      - diffusion step t_idx (B,)
      - condition vector cond (B,C)

    You can swap this MLP with a 1D-Conv/Transformer later.
    """
    def __init__(self, pred_len: int, cond_dim: int, hidden: int = 256, t_emb_dim: int = 128):
        super().__init__()
        self.pred_len = pred_len
        self.t_emb_dim = t_emb_dim

        self.net = nn.Sequential(
            nn.Linear(pred_len + cond_dim + t_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, y_t, t_idx, cond):
        t_emb = sinusoidal_emb(t_idx, self.t_emb_dim).to(y_t.dtype)  # (B, t_emb_dim)
        x = torch.cat([y_t, cond, t_emb], dim=-1)
        return self.net(x)


class LatentODE_DiffusionForecaster(nn.Module):
    def __init__(self, num_features, context_len, pred_len,
                 latent_dim=64, enc_hidden=128, dyn_hidden=128,
                 diff_T=100, diff_hidden=256):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_len * num_features, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, latent_dim),
        )

        self.dynamics = ODEDynamics(latent_dim=latent_dim, hidden=dyn_hidden)

        # Diffusion bits
        self.diffusion = GaussianDiffusion1D(T=diff_T)
        cond_dim = pred_len * latent_dim
        self.eps_head = DiffusionEpsHead(pred_len=pred_len, cond_dim=cond_dim, hidden=diff_hidden)

    def _make_cond(self, z_traj):
        # z_traj: (T,B,D) where T=pred_len
        B = z_traj.shape[1]
        cond = z_traj.transpose(0, 1).reshape(B, self.pred_len * self.latent_dim)
        return cond

    def forward_train(self, x_ctx, y_fut, t_fut):
        """
        Training diffusion loss (epsilon prediction).
        y_fut: (B,p) in standardized space
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        # sample diffusion time
        device = x_ctx.device
        self.diffusion.to(device)
        B = y_fut.shape[0]
        t_idx = torch.randint(0, self.diffusion.T, (B,), device=device, dtype=torch.long)

        y_noisy, eps = self.diffusion.q_sample(y_fut, t_idx)
        eps_hat = self.eps_head(y_noisy, t_idx, cond)
        return torch.nn.functional.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def forward(self, x_ctx, t_fut, n_samples: int = 1):
        """
        Sample future sequences.
        Returns:
          y_hat: (B,p) if n_samples=1 else (n_samples,B,p)
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        device = x_ctx.device
        self.diffusion.to(device)

        B = x_ctx.shape[0]
        P = self.pred_len
        if n_samples == 1:
            return self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device)

        outs = []
        for _ in range(n_samples):
            outs.append(self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device))
        return torch.stack(outs, dim=0)


class KAN_LatentODE_DiffusionForecaster(nn.Module):
    def __init__(self, num_features, context_len, pred_len,
                 latent_dim=64, enc_hidden=128, dyn_hidden=128,
                 diff_T=100, diff_hidden=256):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            KAN([context_len * num_features, enc_hidden]),
            nn.ReLU(),
            KAN([enc_hidden, latent_dim]),
        )

        self.dynamics = ODEDynamics(latent_dim=latent_dim, hidden=dyn_hidden)

        # Diffusion bits
        self.diffusion = GaussianDiffusion1D(T=diff_T)
        cond_dim = pred_len * latent_dim
        self.eps_head = DiffusionEpsHead(pred_len=pred_len, cond_dim=cond_dim, hidden=diff_hidden)

    def _make_cond(self, z_traj):
        # z_traj: (T,B,D) where T=pred_len
        B = z_traj.shape[1]
        cond = z_traj.transpose(0, 1).reshape(B, self.pred_len * self.latent_dim)
        return cond

    def forward_train(self, x_ctx, y_fut, t_fut):
        """
        Training diffusion loss (epsilon prediction).
        y_fut: (B,p) in standardized space
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        # sample diffusion time
        device = x_ctx.device
        self.diffusion.to(device)
        B = y_fut.shape[0]
        t_idx = torch.randint(0, self.diffusion.T, (B,), device=device, dtype=torch.long)

        y_noisy, eps = self.diffusion.q_sample(y_fut, t_idx)
        eps_hat = self.eps_head(y_noisy, t_idx, cond)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def forward(self, x_ctx, t_fut, n_samples: int = 1):
        """
        Sample future sequences.
        Returns:
          y_hat: (B,p) if n_samples=1 else (n_samples,B,p)
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        device = x_ctx.device
        self.diffusion.to(device)

        B = x_ctx.shape[0]
        P = self.pred_len
        if n_samples == 1:
            return self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device)

        outs = []
        for _ in range(n_samples):
            outs.append(self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device))
        return torch.stack(outs, dim=0)


# ----------- Logistic Basis Expansion -----------
class LogisticBasis(nn.Module):
    def __init__(self, in_dim, num_basis):
        super().__init__()
        self.a = nn.Parameter(torch.randn(in_dim, num_basis))
        self.b = nn.Parameter(torch.randn(in_dim, num_basis))

    def forward(self, x):  # x: (B, in_dim)
        x = x.unsqueeze(-1)  # → (B, in_dim, 1)
        return 2 / (1 + torch.exp(-self.a * (x - self.b)))  # (B, in_dim, num_basis)



class LogisticBasisLinear(nn.Module):
    """
    Like a linear layer, but inputs first go through logistic basis expansion.

    x: (B, in_dim)
      -> phi: (B, in_dim*num_basis)
      -> y: (B, out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, num_basis: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.basis = LogisticBasis(in_dim, num_basis)

        self.weight = nn.Parameter(torch.randn(in_dim * num_basis, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.basis(x).reshape(x.shape[0], -1)      # (B, in_dim*num_basis)
        y = phi @ self.weight                             # (B, out_dim)
        if self.bias is not None:
            y = y + self.bias
        return y


# ----------- Fully Nonlinear RNN Cell -----------
class FullyNonlinearKANCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_basis):
        super().__init__()
        self.input_basis = LogisticBasis(input_size, num_basis)
        self.hidden_basis = LogisticBasis(hidden_size, num_basis)
        self.activation = nn.Sigmoid()

        self.hidden_size = hidden_size
        self.num_basis = num_basis

    def forward(self, x_t, h_prev):  # x_t: (B, input_size), h_prev: (B, hidden_size)
        x_phi = self.input_basis(x_t).view(x_t.size(0), -1)       # (B, input_size * num_basis)
        h_phi = self.hidden_basis(h_prev).view(h_prev.size(0), -1)  # (B, hidden_size * num_basis)
        combined = torch.cat([x_phi, h_phi], dim=1)  # (B, total)
        out = self.activation(combined)
        return out[:, :self.hidden_size]  # Truncate back to hidden size


class KANRNNEncoder(nn.Module):
    """
    Uses your FullyNonlinearKANCell repeatedly over context steps,
    then projects final hidden -> latent z0.
    """
    def __init__(self, num_features: int, hidden_size: int, latent_dim: int, num_basis: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = FullyNonlinearKANCell(num_features, hidden_size, num_basis)
        self.to_latent = nn.Linear(hidden_size, latent_dim)

    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        # x_ctx: (B, context_len, F)
        B, T, F = x_ctx.shape
        h = torch.zeros(B, self.hidden_size, device=x_ctx.device, dtype=x_ctx.dtype)

        for t in range(T):
            h = self.rnn_cell(x_ctx[:, t, :], h)  # (B, hidden_size)

        z0 = self.to_latent(h)  # (B, latent_dim)
        return z0



class KAN_FET_LatentODE_DiffusionForecaster(nn.Module):
    def __init__(self, num_features, context_len, pred_len,
                 latent_dim=64,     rnn_hidden=64,
                 num_basis=10, enc_hidden=128, dyn_hidden=128,
                 diff_T=100, diff_hidden=256):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.encoder = KANRNNEncoder(
            num_features=num_features,
            hidden_size=rnn_hidden,
            latent_dim=latent_dim,
            num_basis=num_basis,
        )

        self.dynamics = ODEDynamics(latent_dim=latent_dim, hidden=dyn_hidden)

        # Diffusion bits
        self.diffusion = GaussianDiffusion1D(T=diff_T)
        cond_dim = pred_len * latent_dim
        self.eps_head = DiffusionEpsHead(pred_len=pred_len, cond_dim=cond_dim, hidden=diff_hidden)

    def _make_cond(self, z_traj):
        # z_traj: (T,B,D) where T=pred_len
        B = z_traj.shape[1]
        cond = z_traj.transpose(0, 1).reshape(B, self.pred_len * self.latent_dim)
        return cond

    def forward_train(self, x_ctx, y_fut, t_fut):
        """
        Training diffusion loss (epsilon prediction).
        y_fut: (B,p) in standardized space
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        # sample diffusion time
        device = x_ctx.device
        self.diffusion.to(device)
        B = y_fut.shape[0]
        t_idx = torch.randint(0, self.diffusion.T, (B,), device=device, dtype=torch.long)

        y_noisy, eps = self.diffusion.q_sample(y_fut, t_idx)
        eps_hat = self.eps_head(y_noisy, t_idx, cond)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def forward(self, x_ctx, t_fut, n_samples: int = 1):
        """
        Sample future sequences.
        Returns:
          y_hat: (B,p) if n_samples=1 else (n_samples,B,p)
        """
        z0 = self.encoder(x_ctx)
        z_traj = odeint(self.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = self._make_cond(z_traj)

        device = x_ctx.device
        self.diffusion.to(device)

        B = x_ctx.shape[0]
        P = self.pred_len
        if n_samples == 1:
            return self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device)

        outs = []
        for _ in range(n_samples):
            outs.append(self.diffusion.p_sample_loop(self.eps_head, (B, P), cond, device))
        return torch.stack(outs, dim=0)


@torch.no_grad()
def eval_diffusion_mse(model, diffusion, loader, t_fut, device, n_samples: int = 1):
    """
    For a quick scalar metric, sample once (or a few times) and compute MSE in standardized space.
    """
    model.eval()
    total = 0.0
    n = 0
    for x_ctx, y_fut in loader:
        x_ctx = x_ctx.to(device).float()
        y_fut = y_fut.to(device).float()
        z0 = model.encoder(x_ctx)
        z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = model._make_cond(z_traj)          # (B,D)

        # sample forecast(s)
        y_hat = 0.0
        for _ in range(n_samples):
            y_hat = y_hat + diffusion.p_sample_loop(model.eps_head, y_fut.shape, cond, device=device)
        y_hat = y_hat / float(n_samples)

        total += torch.nn.functional.mse_loss(y_hat, y_fut, reduction="sum").item()
        n += y_fut.numel()
    return total / max(1, n)

def train_and_predict_diffusion(cfg: TrainConfig):
    diff_T = 200
    set_seed(cfg.seed)
    df_raw, df_num, X_raw, y_raw = load_climate_csv(cfg.csv_path, cfg.target_col)
    N, F = X_raw.shape

    train_end, val_end, test_end = split_time_series(N, cfg.val_ratio, cfg.test_ratio)

    # Fit scalers on TRAIN ONLY
    X_mu, X_sd = standardize_fit(X_raw[:train_end])
    y_mu, y_sd = standardize_fit(y_raw[:train_end].reshape(-1, 1))
    y_mu = y_mu.reshape(-1)
    y_sd = y_sd.reshape(-1)

    X = standardize_apply(X_raw, X_mu, X_sd)
    y = standardize_apply(y_raw.reshape(-1, 1), y_mu.reshape(1,1), y_sd.reshape(1,1)).reshape(-1)

    def make_split(start, end):
        return EnergyWindowDataset(X[start:end], y[start:end], cfg.context_len, cfg.pred_len)

    train_ds = make_split(0, train_end)
    val_ds   = make_split(train_end - cfg.context_len - cfg.pred_len + 1, val_end)
    test_ds  = make_split(val_end - cfg.context_len - cfg.pred_len + 1, test_end)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = cfg.device
    model = cfg.model

    diffusion = GaussianDiffusion1D(
        T=diff_T
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # future times for latent ODE (0..pred_len-1)
    t_fut = torch.linspace(0.0, float(cfg.pred_len - 1), steps=cfg.pred_len, device=device)

    train_loss_hist = []
    val_mse_hist = []
    test_mse_hist = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0

        for x_ctx, y_fut in train_loader:
            x_ctx = x_ctx.to(device).float()
            y_fut = y_fut.to(device).float()  # standardized ground truth

            # diffusion training step: sample timestep, add noise, predict noise
            B = x_ctx.size(0)
            t_idx = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(y_fut)
            y_noisy, eps = diffusion.q_sample(y_fut, t_idx, eps)
            z0 = model.encoder(x_ctx)
            z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
            cond = model._make_cond(z_traj)          # (B,D)
            eps_hat = model.eps_head(y_noisy, t_idx, cond)

            loss = torch.nn.functional.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            n_batches += 1

        tr_loss = total / max(1, n_batches)
        train_loss_hist.append(tr_loss)

        # quick eval metrics (sampling-based MSE, standardized)
        va_mse = eval_diffusion_mse(model, diffusion, val_loader, t_fut, device, n_samples=1)
        te_mse = eval_diffusion_mse(model, diffusion, test_loader, t_fut, device, n_samples=1)
        val_mse_hist.append(va_mse)
        test_mse_hist.append(te_mse)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train diff-loss {tr_loss:.5f} | val MSE {va_mse:.5f} | test MSE {te_mse:.5f}")

    # ---- apples-to-apples last AVAILABLE segment prediction (aligned window) ----
    # pick the last valid start index so that future is inside the series:
    last_start = N - (cfg.context_len + cfg.pred_len)
    x_ctx = torch.from_numpy(X[last_start:last_start + cfg.context_len]).unsqueeze(0).to(device).float()  # (1,c,F)
    y_true_std = y[last_start + cfg.context_len : last_start + cfg.context_len + cfg.pred_len]           # (p,) standardized
    y_true = y_true_std * y_sd[0] + y_mu[0]  # de-std

    model.eval()
    with torch.no_grad():
        z0 = model.encoder(x_ctx)
        z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = model._make_cond(z_traj)
        y_hat_std = diffusion.p_sample_loop(model.eps_head, (1, cfg.pred_len), cond, device).squeeze(0).cpu().numpy()
    y_hat = y_hat_std * y_sd[0] + y_mu[0]

    forecast_trend_plot(
        plot_name="Diffusion Neural ODE Climate Forecast vs ground truths",
        model=model,
        X=X,                    # standardized X (NOT X_raw)
        y_raw=y_raw,            # original target units
        context_len=cfg.context_len,
        pred_len=cfg.pred_len,
        t_fut=t_fut,
        y_mu=y_mu,
        y_sd=y_sd,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        device=cfg.device,
        rk4_substeps=cfg.rk4_substeps,
        horizon=cfg.pred_len - 1,
        stride=1,
    )
    return {
        "model": model,
        "diffusion": diffusion,
        "train_diff_loss": np.array(train_loss_hist),
        "val_mse_std": np.array(val_mse_hist),
        "test_mse_std": np.array(test_mse_hist),
        "pred_future": y_hat,
        "true_future": y_true,
        "target_col": cfg.target_col,
        "context_len": cfg.context_len,
        "pred_len": cfg.pred_len,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd,
    }


def train_and_predict_kan_diffusion(cfg: TrainConfig):
    diff_T = 200
    set_seed(cfg.seed)
    df_raw, df_num, X_raw, y_raw = load_climate_csv(cfg.csv_path, cfg.target_col)
    N, F = X_raw.shape

    train_end, val_end, test_end = split_time_series(N, cfg.val_ratio, cfg.test_ratio)

    # Fit scalers on TRAIN ONLY
    X_mu, X_sd = standardize_fit(X_raw[:train_end])
    y_mu, y_sd = standardize_fit(y_raw[:train_end].reshape(-1, 1))
    y_mu = y_mu.reshape(-1)
    y_sd = y_sd.reshape(-1)

    X = standardize_apply(X_raw, X_mu, X_sd)
    y = standardize_apply(y_raw.reshape(-1, 1), y_mu.reshape(1,1), y_sd.reshape(1,1)).reshape(-1)

    def make_split(start, end):
        return EnergyWindowDataset(X[start:end], y[start:end], cfg.context_len, cfg.pred_len)

    train_ds = make_split(0, train_end)
    val_ds   = make_split(train_end - cfg.context_len - cfg.pred_len + 1, val_end)
    test_ds  = make_split(val_end - cfg.context_len - cfg.pred_len + 1, test_end)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = cfg.device
    model = cfg.model.to(device)

    diffusion = GaussianDiffusion1D(
        T=diff_T
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # future times for latent ODE (0..pred_len-1)
    t_fut = torch.linspace(0.0, float(cfg.pred_len - 1), steps=cfg.pred_len, device=device)

    train_loss_hist = []
    val_mse_hist = []
    test_mse_hist = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0

        for x_ctx, y_fut in train_loader:
            x_ctx = x_ctx.to(device).float()
            y_fut = y_fut.to(device).float()  # standardized ground truth

            # diffusion training step: sample timestep, add noise, predict noise
            B = x_ctx.size(0)
            t_idx = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(y_fut)
            y_noisy, eps = diffusion.q_sample(y_fut, t_idx, eps)
            z0 = model.encoder(x_ctx)
            z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
            cond = model._make_cond(z_traj)          # (B,D)
            eps_hat = model.eps_head(y_noisy, t_idx, cond)

            loss = torch.nn.functional.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            n_batches += 1

        tr_loss = total / max(1, n_batches)
        train_loss_hist.append(tr_loss)

        # quick eval metrics (sampling-based MSE, standardized)
        va_mse = eval_diffusion_mse(model, diffusion, val_loader, t_fut, device, n_samples=1)
        te_mse = eval_diffusion_mse(model, diffusion, test_loader, t_fut, device, n_samples=1)
        val_mse_hist.append(va_mse)
        test_mse_hist.append(te_mse)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train diff-loss {tr_loss:.5f} | val MSE {va_mse:.5f} | test MSE {te_mse:.5f}")

    # ---- apples-to-apples last AVAILABLE segment prediction (aligned window) ----
    # pick the last valid start index so that future is inside the series:
    last_start = N - (cfg.context_len + cfg.pred_len)
    x_ctx = torch.from_numpy(X[last_start:last_start + cfg.context_len]).unsqueeze(0).to(device).float()  # (1,c,F)
    y_true_std = y[last_start + cfg.context_len : last_start + cfg.context_len + cfg.pred_len]           # (p,) standardized
    y_true = y_true_std * y_sd[0] + y_mu[0]  # de-std

    model.eval()
    with torch.no_grad():
        z0 = model.encoder(x_ctx)
        z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = model._make_cond(z_traj)
        y_hat_std = diffusion.p_sample_loop(model.eps_head, (1, cfg.pred_len), cond, device).squeeze(0).cpu().numpy()
    y_hat = y_hat_std * y_sd[0] + y_mu[0]

    forecast_trend_plot(
        plot_name="KAN Diffusion Neural ODE Climate Forecast vs ground truths",
        model=model,
        X=X,                    # standardized X (NOT X_raw)
        y_raw=y_raw,            # original target units
        context_len=cfg.context_len,
        pred_len=cfg.pred_len,
        t_fut=t_fut,
        y_mu=y_mu,
        y_sd=y_sd,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        device=cfg.device,
        rk4_substeps=cfg.rk4_substeps,
        horizon=cfg.pred_len - 1,
        stride=1,
    )
    return {
        "model": model,
        "diffusion": diffusion,
        "train_diff_loss": np.array(train_loss_hist),
        "val_mse_std": np.array(val_mse_hist),
        "test_mse_std": np.array(test_mse_hist),
        "pred_future": y_hat,
        "true_future": y_true,
        "target_col": cfg.target_col,
        "context_len": cfg.context_len,
        "pred_len": cfg.pred_len,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd,
    }



def train_and_predict_kan_fet_diffusion(cfg: TrainConfig):
    diff_T = 200
    set_seed(cfg.seed)
    df_raw, df_num, X_raw, y_raw = load_climate_csv(cfg.csv_path, cfg.target_col)
    N, F = X_raw.shape

    train_end, val_end, test_end = split_time_series(N, cfg.val_ratio, cfg.test_ratio)

    # Fit scalers on TRAIN ONLY
    X_mu, X_sd = standardize_fit(X_raw[:train_end])
    y_mu, y_sd = standardize_fit(y_raw[:train_end].reshape(-1, 1))
    y_mu = y_mu.reshape(-1)
    y_sd = y_sd.reshape(-1)

    X = standardize_apply(X_raw, X_mu, X_sd)
    y = standardize_apply(y_raw.reshape(-1, 1), y_mu.reshape(1,1), y_sd.reshape(1,1)).reshape(-1)

    def make_split(start, end):
        return EnergyWindowDataset(X[start:end], y[start:end], cfg.context_len, cfg.pred_len)

    train_ds = make_split(0, train_end)
    val_ds   = make_split(train_end - cfg.context_len - cfg.pred_len + 1, val_end)
    test_ds  = make_split(val_end - cfg.context_len - cfg.pred_len + 1, test_end)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = cfg.device
    model = cfg.model.to(device)
    diffusion = GaussianDiffusion1D(
        T=diff_T
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # future times for latent ODE (0..pred_len-1)
    t_fut = torch.linspace(0.0, float(cfg.pred_len - 1), steps=cfg.pred_len, device=device)

    train_loss_hist = []
    val_mse_hist = []
    test_mse_hist = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0

        for x_ctx, y_fut in train_loader:
            x_ctx = x_ctx.to(device).float()
            y_fut = y_fut.to(device).float()  # standardized ground truth

            # diffusion training step: sample timestep, add noise, predict noise
            B = x_ctx.size(0)
            t_idx = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(y_fut)
            y_noisy, eps = diffusion.q_sample(y_fut, t_idx, eps)
            z0 = model.encoder(x_ctx)
            z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
            cond = model._make_cond(z_traj)          # (B,D)
            eps_hat = model.eps_head(y_noisy, t_idx, cond)

            loss = torch.nn.functional.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            n_batches += 1

        tr_loss = total / max(1, n_batches)
        train_loss_hist.append(tr_loss)

        # quick eval metrics (sampling-based MSE, standardized)
        va_mse = eval_diffusion_mse(model, diffusion, val_loader, t_fut, device, n_samples=1)
        te_mse = eval_diffusion_mse(model, diffusion, test_loader, t_fut, device, n_samples=1)
        val_mse_hist.append(va_mse)
        test_mse_hist.append(te_mse)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | train diff-loss {tr_loss:.5f} | val MSE {va_mse:.5f} | test MSE {te_mse:.5f}")

    # ---- apples-to-apples last AVAILABLE segment prediction (aligned window) ----
    # pick the last valid start index so that future is inside the series:
    last_start = N - (cfg.context_len + cfg.pred_len)
    x_ctx = torch.from_numpy(X[last_start:last_start + cfg.context_len]).unsqueeze(0).to(device).float()  # (1,c,F)
    y_true_std = y[last_start + cfg.context_len : last_start + cfg.context_len + cfg.pred_len]           # (p,) standardized
    y_true = y_true_std * y_sd[0] + y_mu[0]  # de-std

    model.eval()
    with torch.no_grad():
        z0 = model.encoder(x_ctx)
        z_traj = odeint(model.dynamics, z0, t_fut, method="dopri5")  # (T,B,D)
        cond = model._make_cond(z_traj)
        y_hat_std = diffusion.p_sample_loop(model.eps_head, (1, cfg.pred_len), cond, device).squeeze(0).cpu().numpy()
    y_hat = y_hat_std * y_sd[0] + y_mu[0]

    forecast_trend_plot(
        plot_name="KAN-FET Diffusion Neural ODE Climate Forecast vs ground truths",
        model=model,
        X=X,                    # standardized X (NOT X_raw)
        y_raw=y_raw,            # original target units
        context_len=cfg.context_len,
        pred_len=cfg.pred_len,
        t_fut=t_fut,
        y_mu=y_mu,
        y_sd=y_sd,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        device=cfg.device,
        rk4_substeps=cfg.rk4_substeps,
        horizon=cfg.pred_len - 1,
        stride=1,
    )
    return {
        "model": model,
        "diffusion": diffusion,
        "train_diff_loss": np.array(train_loss_hist),
        "val_mse_std": np.array(val_mse_hist),
        "test_mse_std": np.array(test_mse_hist),
        "pred_future": y_hat,
        "true_future": y_true,
        "target_col": cfg.target_col,
        "context_len": cfg.context_len,
        "pred_len": cfg.pred_len,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd,
    }



if __name__ == "__main__":

    EPOCHS = 100

    cfg = TrainConfig(
        model= LatentNeuralODEForecaster(
            num_features=6,          # Climate.csv has 6 numeric columns
            context_len=50,
            pred_len=12,
            latent_dim=64,
        ).to("cuda" if torch.cuda.is_available() else "cpu"),
        csv_path="Time_MMD/numerical/Climate/Climate.csv",
        target_col="OT",
        context_len=50,
        pred_len=12,
        batch_size=48,
        epochs=EPOCHS,
        lr=1.5e-3,
        weight_decay=1e-2,
        latent_dim=64,
    )

   

    out = train_and_predict(cfg)

    print("\nMLP-Style NODE Forecast (next steps) for", out["target_col"])
    print(out["pred_future"])
    if out["true_future"] is not None:
        print("\nLast available ground truth (for reference):")
        print(out["true_future"])

    
    diffusion_cfg = TrainConfig(
        model = LatentODE_DiffusionForecaster(
        num_features=6,
        context_len=50,
        pred_len=12,
        latent_dim=64,
        ),  # will be created inside
        csv_path="Time_MMD/numerical/Climate/Climate.csv",
        target_col="OT",
        context_len=50,
        pred_len=12,
        batch_size=64,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        latent_dim=64,
    )
    diffusion_out = train_and_predict_diffusion(diffusion_cfg)


    print("\nDiffusion ODE Model Forecast (next steps) for", diffusion_out["target_col"])
    print(diffusion_out["pred_future"])
    if diffusion_out["true_future"] is not None:
        print("\nLast available ground truth (for reference):")
        print(diffusion_out["true_future"])

    kan_diffusion_cfg = TrainConfig(
        model = KAN_LatentODE_DiffusionForecaster(
        num_features=6,
        context_len=50,
        pred_len=12,    
        latent_dim=64),
        csv_path="Time_MMD/numerical/Climate/Climate.csv",
        target_col="OT",
        context_len=50,
        pred_len=12,
        batch_size=64,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        latent_dim=64,
    )
    kan_diffusion_out = train_and_predict_kan_diffusion(kan_diffusion_cfg)

    print("\nKAN Diffusion ODE Model Forecast (next steps) for", kan_diffusion_out["target_col"])
    print(kan_diffusion_out["pred_future"])
    if kan_diffusion_out["true_future"] is not None:
        print("\nLast available ground truth (for reference):")
        print(kan_diffusion_out["true_future"])

     
    kan_fet_diffusion_cfg = TrainConfig(
        model = KAN_FET_LatentODE_DiffusionForecaster(
        num_features=6,
        context_len=50,
        pred_len=12,
        latent_dim=64),
        csv_path="Time_MMD/numerical/Climate/Climate.csv",
        target_col="OT",
        context_len=50,
        pred_len=12, 
        batch_size=64,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        latent_dim=64,
    )
    kan_fet_diffusion_out = train_and_predict_kan_fet_diffusion(kan_fet_diffusion_cfg)
    print("\nKAN-FET Diffusion ODE Model Forecast (next steps) for", kan_fet_diffusion_out["target_col"])
    print(kan_fet_diffusion_out["pred_future"])
    if kan_fet_diffusion_out["true_future"] is not None:
        print("\nLast available ground truth (for reference):")
        print(kan_fet_diffusion_out["true_future"])
    plot_train_val_loss(out, diffusion_out, kan_diffusion_out, kan_fet_diffusion_out)
    