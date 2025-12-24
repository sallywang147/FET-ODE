#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Diffusion for Time-Series Forecasting (ETT-style datasets)

Goal:
- Given past seq_len window (seq_x), generate/forecast future pred_len window (future_y)
  by training a conditional diffusion model that denoises noisy future conditioned on past.

Works with Dataset_ETT_hour / Dataset_ETT_minute / Dataset_Custom from your code.

Notes:
- Uses num_workers=0 by default (safe on macOS / spawn).
- If `from kan import KAN` is available, you can enable a KAN-based backbone with --backbone kan
  (fallbacks to MLP if import fails).
- This is a single runnable .py file.

"""

import os
import math
import argparse
from kan_diffusion.kan import KAN
from dataclasses import dataclass
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficient_kan.efficientkan import KANFET
from torchdiffeq import odeint
import matplotlib.pyplot as plt


# -------------------------
# Your dataset classes import
# -------------------------
# If this script lives alongside your dataset file, import them here.
# Otherwise paste your Dataset_ETT_hour/Dataset_ETT_minute/Dataset_Custom classes above this script
# and delete the import section below.

# ---- BEGIN: minimal safe import pattern ----
try:
    # Change this if your dataset file/module name differs
    from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
except Exception:
    Dataset_ETT_hour = None
    Dataset_ETT_minute = None
    Dataset_Custom = None
# ---- END: import pattern ----


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"



def plot_loss(
    step, loss,
    step_eft, loss_eft,
    step_a, loss_a,
    step_b, loss_b,
    step_c, loss_c,
    title="Training Loss",
):
    plt.figure(figsize=(10, 6))

    plt.plot(step, loss, label="KAN-ODE Diffusion", linewidth=2)
    plt.plot(step_eft, loss_eft, label="KAN-FET-ALL-NODE Diffusion", linewidth=2, linestyle="-.")
    plt.plot(step_a, loss_a, label="KAN Diffusion", linewidth=2, linestyle="--")
    plt.plot(step_b, loss_b, label="MLP Diffusion", linewidth=2, linestyle=":")
    plt.plot(step_c, loss_c, label="KAN-FET-LINEAR-NODE Diffusion", linewidth=2, linestyle="-.")
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(title, fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(
        loc="best",          # automatically choose a non-overlapping spot
        fontsize=12,
        frameon=True,
        framealpha=0.9,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()






# -------------------------
# Diffusion schedule
# -------------------------
@dataclass
class DiffusionSchedule:
    T: int = 250
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def make(self, device):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)  # (T,)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        sqrt_abar = torch.sqrt(abar)
        sqrt_1m_abar = torch.sqrt(1.0 - abar)
        return betas, alphas, abar, sqrt_abar, sqrt_1m_abar


# -------------------------
# Time embedding
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: (B,) integer timesteps in [0..T-1]
        returns: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb



def _interp_1d_batch(x_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation over sequence dimension for a batch.

    x_seq: (B, L, D)
    t: scalar tensor in [0,1] (or python float) OR shape () tensor
    returns: (B, D)
    """
    B, L, D = x_seq.shape

    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x_seq.device, dtype=x_seq.dtype)
    t = t.to(device=x_seq.device, dtype=x_seq.dtype)

    # map t in [0,1] -> position in [0, L-1]
    pos = t.clamp(0.0, 1.0) * (L - 1)
    i0 = torch.floor(pos).long().clamp(0, L - 1)
    i1 = (i0 + 1).clamp(0, L - 1)
    w = (pos - i0.to(pos.dtype)).view(1, 1)  # broadcast over (B, D)

    x0 = x_seq[:, i0, :]  # (B, D)
    x1 = x_seq[:, i1, :]  # (B, D)
    return (1.0 - w) * x0 + w * x1




class PastODEFunc(nn.Module):
    """
    dz/dt = f(z, x(t))
    """
    def __init__(self, z_dim: int, x_dim: int, hidden: int = 128, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(z_dim) if use_ln else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        self._x_seq = None  # (B, L, x_dim)

    def set_signal(self, x_seq: torch.Tensor):
        self._x_seq = x_seq

    def forward(self, t, z):
        assert self._x_seq is not None, "Call odefunc.set_signal(x_seq) before odeint."
        x_t = _interp_1d_batch(self._x_seq, t)  # (B, x_dim)
        z = self.ln(z)
        return self.net(torch.cat([z, x_t], dim=-1))


# -------------------------
# Conditional denoiser model
# -------------------------
class PastEncoder(nn.Module):
    """
    Encode past seq_x: (B, Lx, D) -> (B, C)
    """
    def __init__(self, d_in: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, past):
        # past: (B, Lx, D) -> conv expects (B, D, Lx)
        x = past.transpose(1, 2)
        h = self.net(x).squeeze(-1)  # (B, hidden)
        return self.proj(h)          # (B, out_dim)


class MLPDenoiser(nn.Module):
    """
    Predict noise eps for future sequence x_t given conditioning on past and timestep t.

    Inputs:
      x_t:   (B, Ly, D)  noisy future
      past:  (B, Lx, D)  conditioning past
      t:     (B,)        timestep
    Output:
      eps_hat: (B, Ly, D)
    """
    def __init__(self, d_in: int, pred_len: int, cond_dim: int = 128, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_in = d_in

        self.past_enc = PastEncoder(d_in=d_in, hidden=128, out_dim=cond_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = pred_len * d_in + cond_dim + time_dim
        out_dim = pred_len * d_in

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_t, past, t):
        B = x_t.shape[0]
        cond = self.past_enc(past)           # (B, cond_dim)
        temb = self.time_emb(t)              # (B, time_dim)

        x_flat = x_t.reshape(B, -1)          # (B, Ly*D)
        h = torch.cat([x_flat, cond, temb], dim=1)
        eps = self.net(h).reshape(B, self.pred_len, self.d_in)
        return eps



class PastEncoderNeuralODE(nn.Module):
    """
    Encode past: (B, L, d_in) -> (B, cond_dim) using a conditional Neural ODE.
    """
    def __init__(
        self,
        d_in: int,
        cond_dim: int = 128,
        x_proj_dim: int | None = None,
        ode_hidden: int = 128,
        dropout: float = 0.0,
        n_eval: int = 5,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.d_in = d_in
        self.cond_dim = cond_dim
        self.x_proj_dim = x_proj_dim or cond_dim
        self.n_eval = n_eval
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Project x(t) so the ODE sees a richer / consistent dimension.
        self.x_proj = nn.Linear(d_in, self.x_proj_dim)

        self.odefunc = PastODEFunc(
            z_dim=cond_dim,
            x_dim=self.x_proj_dim,     # ✅ REQUIRED
            hidden=ode_hidden,
            dropout=dropout,
            use_ln=True,
        )

        # z0 from first observation (or mean). Keeping it simple:
        self.z0_proj = nn.Linear(self.x_proj_dim, cond_dim)

    def forward(self, past: torch.Tensor):
        """
        past: (B, L, d_in)
        returns cond: (B, cond_dim)
        """
        from torchdiffeq import odeint  # or torchode/torchodeint depending on your stack

        B, L, D = past.shape
        x_seq = self.x_proj(past)              # (B, L, x_proj_dim)
        self.odefunc.set_signal(x_seq)

        z0 = self.z0_proj(x_seq[:, 0, :])      # (B, cond_dim)

        # Integrate over [0,1]
        t = torch.linspace(0.0, 1.0, self.n_eval, device=past.device, dtype=past.dtype)
        zt = odeint(self.odefunc, z0, t, method=self.solver, rtol=self.rtol, atol=self.atol)
        zT = zt[-1]                            # (B, cond_dim)
        return zT


class KAN_NODE(nn.Module):
    def __init__(self, d_in: int, pred_len: int, cond_dim: int = 128, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_in = d_in

        # ✅ Construct the module ONCE
        self.past_enc = PastEncoderNeuralODE(
            d_in=d_in,
            cond_dim=cond_dim,
            ode_hidden=128,
            dropout=0.0,
            n_eval=5,
        )

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = pred_len * d_in + cond_dim + time_dim
        out_dim = pred_len * d_in

        try:
            from kan import KAN
            self.net = nn.Sequential(
                KAN([in_dim, hidden]),
                KAN([hidden, hidden]),
                KAN([hidden, out_dim]),
            )
            self._is_kan = True
        except Exception:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            self._is_kan = False

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_t, past, t):
        """
        x_t:  (B, pred_len, d_in)
        past: (B, seq_len, d_in)
        t:    (B,) or scalar-like (depends on your time embedding)
        """
        B = x_t.shape[0]
        cond = self.past_enc(past)          # ✅ tensor (B, cond_dim)
        temb = self.time_emb(t)             # (B, time_dim)
        x_flat = x_t.reshape(B, -1)
        h = torch.cat([x_flat, cond, temb], dim=1)
        eps = self.net(h).reshape(B, self.pred_len, self.d_in)
        return eps


class KAN_FET_ALL_NODE(nn.Module):
    def __init__(self, d_in: int, pred_len: int, cond_dim: int = 128, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_in = d_in

        # ✅ Construct the module ONCE
        self.past_enc = PastEncoderNeuralODE(
            d_in=d_in,
            cond_dim=cond_dim,
            ode_hidden=128,
            dropout=0.0,
            n_eval=5,
        )

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = pred_len * d_in + cond_dim + time_dim
        out_dim = pred_len * d_in
        self.net = nn.Sequential(
            KANFET([in_dim, hidden]),
            KANFET([hidden, hidden]),
            KANFET([hidden, out_dim]),
            )
      
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_t, past, t):
        """
        x_t:  (B, pred_len, d_in)
        past: (B, seq_len, d_in)
        t:    (B,) or scalar-like (depends on your time embedding)
        """
        B = x_t.shape[0]
        cond = self.past_enc(past)          # ✅ tensor (B, cond_dim)
        temb = self.time_emb(t)             # (B, time_dim)
        x_flat = x_t.reshape(B, -1)
        h = torch.cat([x_flat, cond, temb], dim=1)
        eps = self.net(h).reshape(B, self.pred_len, self.d_in)
        return eps
    
class KANDenoiser(nn.Module):
    """
    Same interface as MLPDenoiser, but uses KAN blocks if available.
    Falls back to MLP if KAN import fails.
    """
    def __init__(self, d_in: int, pred_len: int, cond_dim: int = 128, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_in = d_in

        self.past_enc = PastEncoder(d_in=d_in, hidden=128, out_dim=cond_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = pred_len * d_in + cond_dim + time_dim
        out_dim = pred_len * d_in

        # Try to use KAN; if not installed, use MLP
        try:
            from kan import KAN  # pip/your repo
            self.net = nn.Sequential(
                KAN([in_dim, hidden]),
                KAN([hidden, hidden]),
                KAN([hidden, out_dim]),
            )
            self._is_kan = True
        except Exception:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            self._is_kan = False

        # Initialize Linear layers if present
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_t, past, t):
        B = x_t.shape[0]
        cond = self.past_enc(past)
        temb = self.time_emb(t)
        x_flat = x_t.reshape(B, -1)
        h = torch.cat([x_flat, cond, temb], dim=1)
        eps = self.net(h).reshape(B, self.pred_len, self.d_in)
        return eps


class KAN_FET_LINEAR_ODE(nn.Module):
    """
    Same interface as MLPDenoiser, but uses KAN blocks if available.
    Falls back to MLP if KAN import fails.
    """
    def __init__(self, d_in: int, pred_len: int, cond_dim: int = 128, time_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_in = d_in

        self.past_enc = PastEncoder(d_in=d_in, hidden=128, out_dim=cond_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = pred_len * d_in + cond_dim + time_dim
        out_dim = pred_len * d_in

        # Try to use KAN; if not installed, use MLP
        try:
            from kan import KAN  # pip/your repo
            self.net = nn.Sequential(
                KANFET([in_dim, hidden]),
                KANFET([hidden, hidden]),
                KANFET([hidden, out_dim]),
            )
            self._is_kan = True
        except Exception:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            self._is_kan = False

        # Initialize Linear layers if present
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_t, past, t):
        B = x_t.shape[0]
        cond = self.past_enc(past)
        temb = self.time_emb(t)
        x_flat = x_t.reshape(B, -1)
        h = torch.cat([x_flat, cond, temb], dim=1)
        eps = self.net(h).reshape(B, self.pred_len, self.d_in)
        return eps


# -------------------------
# DDPM forward/reverse
# -------------------------
def q_sample(y0, t, sqrt_abar, sqrt_1m_abar, noise=None):
    """
    y_t = sqrt(alpha_bar_t) * y0 + sqrt(1-alpha_bar_t) * eps
    y0: (B, Ly, D)
    t:  (B,) integer
    """
    if noise is None:
        noise = torch.randn_like(y0)
    B = y0.shape[0]
    a = sqrt_abar[t].view(B, 1, 1)
    b = sqrt_1m_abar[t].view(B, 1, 1)
    return a * y0 + b * noise, noise


@torch.no_grad()
def p_sample_loop(model, past, schedule_tensors, pred_len, d_in):
    """
    Generate y0 ~ p(y|past) by reverse diffusion.
    past: (B, Lx, D)
    returns y0_hat: (B, Ly, D)
    """
    betas, alphas, abar, sqrt_abar, sqrt_1m_abar = schedule_tensors
    device = past.device
    T = betas.shape[0]

    B = past.shape[0]
    y = torch.randn(B, pred_len, d_in, device=device)

    for ti in reversed(range(T)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)
        eps_hat = model(y, past, t)

        # Predict x0 (here y0) from current y_t
        a_bar = abar[ti]
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_1m_a_bar = torch.sqrt(1.0 - a_bar)
        y0_hat = (y - sqrt_1m_a_bar * eps_hat) / (sqrt_a_bar + 1e-8)

        # DDPM mean
        beta_t = betas[ti]
        alpha_t = alphas[ti]
        a_bar_prev = abar[ti - 1] if ti > 0 else torch.tensor(1.0, device=device)

        # posterior variance (beta_tilde)
        beta_tilde = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar + 1e-8)

        # posterior mean coefficients
        coef1 = (torch.sqrt(a_bar_prev) * beta_t) / (1.0 - a_bar + 1e-8)
        coef2 = (torch.sqrt(alpha_t) * (1.0 - a_bar_prev)) / (1.0 - a_bar + 1e-8)

        mean = coef1 * y0_hat + coef2 * y

        if ti > 0:
            z = torch.randn_like(y)
            y = mean + torch.sqrt(beta_tilde + 1e-8) * z
        else:
            y = mean

    return y


# -------------------------
# Dataset adapter for forecasting
# -------------------------
def pick_dataset(args):
    if args.dataset == "ett_hour":
        assert Dataset_ETT_hour is not None, "Failed to import Dataset_ETT_hour. Fix import path."
        cls = Dataset_ETT_hour
    elif args.dataset == "ett_minute":
        assert Dataset_ETT_minute is not None, "Failed to import Dataset_ETT_minute. Fix import path."
        cls = Dataset_ETT_minute
    elif args.dataset == "custom":
        assert Dataset_Custom is not None, "Failed to import Dataset_Custom. Fix import path."
        cls = Dataset_Custom
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    size = (args.seq_len, args.label_len, args.pred_len)
    train_ds = cls(
        root_path=args.root_path,
        flag="train",
        size=size,
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=args.timeenc,
        freq=args.freq,
        cols=None,
    )
    val_ds = cls(
        root_path=args.root_path,
        flag="val",
        size=size,
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=args.timeenc,
        freq=args.freq,
        cols=None,
    )
    test_ds = cls(
        root_path=args.root_path,
        flag="test",
        size=size,
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        scale=True,
        inverse=False,
        timeenc=args.timeenc,
        freq=args.freq,
        cols=None,
    )
    return train_ds, val_ds, test_ds


def extract_future(seq_y, label_len, pred_len):
    """
    In ETT loaders, seq_y is (label_len + pred_len, D). We only forecast the last pred_len.
    """
    return seq_y[label_len:label_len + pred_len]


# -------------------------
# Train / eval
# -------------------------
def train_conditional_diffusion(
    model,
    train_loader,
    val_loader,
    schedule: DiffusionSchedule,
    pred_len: int,
    d_in: int,
    device: str,
    epochs: int = 10,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
    log_every: int = 200,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    betas, alphas, abar, sqrt_abar, sqrt_1m_abar = schedule.make(device)
    schedule_tensors = (betas, alphas, abar, sqrt_abar, sqrt_1m_abar)

    loss_list = []
    val_loss_list = []
    step_list = []
    global_step = 0

    def run_val():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                seq_x, seq_y, _, _ = batch
                seq_x = seq_x.to(device).float()  # (B, Lx, D)
                seq_y = seq_y.to(device).float()  # (B, Ly_total, D)
                future = torch.stack([extract_future(y, args.label_len, pred_len) for y in seq_y], dim=0)
                # future: (B, pred_len, D)

                B = future.shape[0]
                t = torch.randint(0, schedule.T, (B,), device=device).long()
                y_t, noise = q_sample(future, t, sqrt_abar, sqrt_1m_abar, noise=None)
                eps_hat = model(y_t, seq_x, t)
                losses.append(F.mse_loss(eps_hat, noise).item())
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            seq_x, seq_y, _, _ = batch
            seq_x = seq_x.to(device).float()
            seq_y = seq_y.to(device).float()

            future = torch.stack([extract_future(y, args.label_len, pred_len) for y in seq_y], dim=0)

            B = future.shape[0]
            t = torch.randint(0, schedule.T, (B,), device=device).long()

            y_t, noise = q_sample(future, t, sqrt_abar, sqrt_1m_abar, noise=None)
            eps_hat = model(y_t, seq_x, t)

            loss = F.mse_loss(eps_hat, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_list.append(loss.item())
            step_list.append(global_step)
            val_loss = run_val()
            val_loss_list.append(val_loss)
            global_step += 1

            if (global_step % log_every) == 0:
                val_loss = run_val()
                print(f"[ep {ep:03d}] step {global_step:06d}  train_mse={loss.item():.6f}  val_mse={val_loss:.6f}")

        # epoch-end summary
        val_loss = run_val()
        print(f"==> Epoch {ep}/{epochs} done. val_mse={val_loss:.6f}")

    return loss_list, val_loss_list, step_list, model, schedule_tensors


@torch.no_grad()
def evaluate_forecast_samples(model, test_loader, schedule_tensors, pred_len, d_in, device, num_batches=1):
    """
    Quick qualitative plot: past, true future, sampled future.
    """
    model.eval()
    batches = 0
    for batch in test_loader:
        seq_x, seq_y, _, _ = batch
        seq_x = seq_x.to(device).float()
        seq_y = seq_y.to(device).float()

        future_true = torch.stack([extract_future(y, args.label_len, pred_len) for y in seq_y], dim=0)
        future_samp = p_sample_loop(model, seq_x, schedule_tensors, pred_len=pred_len, d_in=d_in)

        # Plot first sample, first dimension
        past_0 = seq_x[0].detach().cpu().numpy()         # (Lx, D)
        true_0 = future_true[0].detach().cpu().numpy()   # (Ly, D)
        samp_0 = future_samp[0].detach().cpu().numpy()   # (Ly, D)

        d0 = 0
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(past_0.shape[0]), past_0[:, d0], label="past", linewidth=2)
        base = past_0.shape[0]
        plt.plot(np.arange(base, base + true_0.shape[0]), true_0[:, d0], label="true_future", linewidth=2)
        plt.plot(np.arange(base, base + samp_0.shape[0]), samp_0[:, d0], label="sampled_future", linewidth=2)
        plt.title("Conditional Diffusion Forecast (one sample, dim=0)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

        batches += 1
        if batches >= num_batches:
            break


@torch.no_grad()
def evaluate_forecast_loss(
    model,
    test_loader,
    schedule_tensors,
    pred_len,
    d_in,
    device,
    label_len,
    loss_type="mse",          # "mse" | "mae" | "rmse"
    num_samples=1,            # number of diffusion samples per input
    max_batches=10,         # None = full test set
):
    """
    Quantitative forecast evaluation using diffusion sampling.

    Returns:
        avg_loss (float)
    """
    model.eval()

    total_loss = 0.0
    total_count = 0

    for b_idx, batch in enumerate(test_loader):
        seq_x, seq_y, _, _ = batch
        seq_x = seq_x.to(device).float()        # (B, Lx, D)
        seq_y = seq_y.to(device).float()        # (B, label_len + pred_len, D)

        # Ground-truth future
        future_true = seq_y[:, label_len:label_len + pred_len, :]  # (B, Ly, D)

        B = future_true.shape[0]

        # Multiple diffusion samples → average prediction
        future_preds = []
        for _ in range(num_samples):
            future_samp = p_sample_loop(
                model,
                seq_x,
                schedule_tensors,
                pred_len=pred_len,
                d_in=d_in,
            )                                   # (B, Ly, D)
            future_preds.append(future_samp)
        
        future_pred = torch.stack(future_preds, dim=0).mean(dim=0)  # (B, Ly, D)
        # Compute loss
        if loss_type == "mse":
            loss = F.mse_loss(future_pred, future_true, reduction="sum")
        elif loss_type == "mae":
            loss = F.l1_loss(future_pred, future_true, reduction="sum")
        elif loss_type == "rmse":
            loss = torch.sqrt(F.mse_loss(future_pred, future_true, reduction="sum"))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        total_loss += loss.item()
        total_count += future_true.numel()
        print(f"loss so far {total_loss / total_count:.6f} on {total_count} samples")
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / total_count
    return avg_loss

# -------------------------
# Main
# -------------------------
def default_config():
    return {
        # ---------------- dataset ----------------
        "dataset": "ett_hour",          # "ett_hour" | "ett_minute" | "custom"
        "root_path": ".",               # root containing the CSV or its parent
        "data_path": "data/ETT/ETTh1.csv",
        "features": "S",                # "S" | "M" | "MS"
        "target": "OT",
        "freq": "h",
        "timeenc": 0,

        # ---------------- windows ----------------
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,

        # ---------------- training ----------------
        "batch_size": 64,
        "epochs": 10,
        "lr": 2e-4,
        "num_workers": 0,               # IMPORTANT for macOS

        # ---------------- diffusion ----------------
        "T": 250,
        "beta_start": 1e-4,
        "beta_end": 0.02,

        # ---------------- model ----------------
        "backbone": "kan",               # "mlp" | "kan"
        "hidden": 256,
        "cond_dim": 128,
        "time_dim": 128,

        # ---------------- misc ----------------
        "seed": 0,
        "log_every": 200,
    }




if __name__ == "__main__":
    # ---------------- default setup ----------------
    cfg = default_config()

    set_seed(cfg["seed"])
    device = default_device()
    print("Device:", device)

    # ---------------- dataset ----------------
    class Args:
        pass
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    train_ds, val_ds, test_ds = pick_dataset(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )

    # ---------------- infer data dims ----------------
    seq_x, seq_y, _, _ = next(iter(train_loader))
    d_in = seq_x.shape[-1]
    print(f"Batch shapes: seq_x={tuple(seq_x.shape)} seq_y={tuple(seq_y.shape)}")

    # ---------------- diffusion schedule ----------------
    schedule = DiffusionSchedule(
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    # ---------------- model ----------------
    kan_ode_model = KAN_NODE(
        d_in=d_in,
        pred_len=args.pred_len,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        hidden=args.hidden,
    )
    print("Model: Conditional KAN-ODE Diffusion")

    kan_eft_ode_model = KAN_FET_ALL_NODE(
        d_in=d_in,
        pred_len=args.pred_len,
        cond_dim=args.cond_dim, 
        time_dim=args.time_dim,
        hidden=args.hidden,
    )
    print("Model: Conditional KAN-FET-ODE Diffusion")
    kan_eft_ode_linear_model = KAN_FET_LINEAR_ODE(
        d_in=d_in,
        pred_len=args.pred_len,
        cond_dim=args.cond_dim, 
        time_dim=args.time_dim,
        hidden=args.hidden,
    )
    print("Model: Conditional KAN-FET-ODE Linear Diffusion")
    kan_model = KANDenoiser(
        d_in=d_in,
        pred_len=args.pred_len,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        hidden=args.hidden,
    )
    print("Model: Conditional KAN Diffusion")
 
    nn_model = MLPDenoiser(
        d_in=d_in,
        pred_len=args.pred_len,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        hidden=args.hidden,
        )
    print("Model: Conditional MLP Diffusion")
    
    # ---------------- train ----------------
    loss, val_loss, step, trained_kan_ode_model, schedule_tensors = train_conditional_diffusion(
        model=kan_ode_model,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=schedule,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=1.0,
        log_every=args.log_every,
    )

    loss_eft, val_loss_eft, step_eft, trained_kan_eft_ode_model, schedule_tensors = train_conditional_diffusion(
        model=kan_eft_ode_model,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=schedule,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=1.0,
        log_every=args.log_every,
    )
    loss_a, val_loss_a, step_a, trained_kan_model, schedule_tensors = train_conditional_diffusion(
        model=kan_model,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=schedule,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=1.0,
        log_every=args.log_every,
    )
    
    loss_b, val_loss_b, step_b, trained_nn_model, schedule_tensors = train_conditional_diffusion(
        model=nn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=schedule,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=1.0,
        log_every=args.log_every,
    )
    
    loss_c, val_loss_c, step_c, trained_kan_eft_ode_linear_model, schedule_tensors = train_conditional_diffusion(
        model=kan_eft_ode_linear_model,
        train_loader=train_loader,  # train_loader=train_loader,
        val_loader=val_loader,  # val_loader=val_loader,
        schedule=schedule,  # schedule=schedule,                    
        pred_len=args.pred_len,     # pred_len=args.pred_len,
        d_in=d_in,                # d_in=d_in,
        device=device,                # device=device,
        epochs=args.epochs,            # epochs=args.epochs,
        lr=args.lr,                # lr=args.lr,
        grad_clip=1.0,            # grad_clip=1.0,
        log_every=args.log_every,    # log_every=args.log_every,
    )
    # ---------------- plots ----------------
    plot_loss(step, loss, step_eft, loss_eft, step_a, loss_a, step_b, loss_b, step_c, loss_c, title=f"Training Loss")

    plot_loss(step, val_loss, step_eft, val_loss_eft, step_a, val_loss_a, step_b, val_loss_b,  step_c, val_loss_c, title=f"Validation Loss")
    # ---------------- qualitative forecast ----------------
    '''
    evaluate_forecast_samples(
        model=trained_kan_ode_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        num_batches=1,
    )

    evaluate_forecast_samples(
        model=trained_kan_eft_ode_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        num_batches=1,
    )

    evaluate_forecast_samples(
        model=trained_kan_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        num_batches=1,
    )

    evaluate_forecast_samples(
        model=trained_nn_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        num_batches=1,
    )
    ''' 
    mlp_test_mse = evaluate_forecast_loss(
        model=trained_nn_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        label_len=args.label_len,
        loss_type="mse",
        num_samples=10,          # diffusion uncertainty averaging
        max_batches=1,         # None = full test set
    )

    
    kan_test_mse = evaluate_forecast_loss(
        model=trained_kan_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        label_len=args.label_len,
        loss_type="mse",
        num_samples=10,
        max_batches=1,
    )


    kan_ode_test_mse = evaluate_forecast_loss(
        model=trained_kan_ode_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,
        d_in=d_in,
        device=device,
        label_len=args.label_len,
        loss_type="mse",
        num_samples=10,          # diffusion uncertainty averaging
        max_batches=1,         # None = full test set
    )
    
    kan_eft_ode_test_mse = evaluate_forecast_loss(
        model=trained_kan_eft_ode_model,
        test_loader=test_loader,
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,     d_in=d_in, 
        device=device,
        label_len=args.label_len,   
        loss_type="mse",
        num_samples=10,          # diffusion uncertainty averaging
        max_batches=1,         # None = full test set
    )
    kan_eft_ode_linear_model_mse = evaluate_forecast_loss(
        model=trained_kan_eft_ode_linear_model,
        test_loader=test_loader,                    
        schedule_tensors=schedule_tensors,
        pred_len=args.pred_len,     # d_in=d_in,
        d_in=d_in,
        device=device,
        label_len=args.label_len,
        loss_type="mse",
        num_samples=10,          # diffusion uncertainty averaging
        max_batches=1,         # None = full test set
    )
    print(f"[MLP] Test Forecast MSE: {mlp_test_mse:.6f}")
    print(f"[KAN] Test Forecast MSE: {kan_test_mse:.6f}")
    print(f"[KAN-ODE] Test Forecast MSE: {kan_ode_test_mse:.6f}")
    print(f"[KAN-FET-ALL-NODE] Test Forecast MSE: {kan_eft_ode_test_mse:.6f}")
    print(f"[KAN-FET-LINEAR-NODE] Test Forecast MSE: {kan_eft_ode_linear_model_mse:.6f}")


