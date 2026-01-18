import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# pip install torchdiffeq
from torchdiffeq import odeint
from kan_diffusion import kan

#Experient note for future self: kan doesn't compose well with linear/MLP with drop out or extra activation
#If we want to combine them, do it at the end with a single linear layer usually works.
# =========================
# Dataset (yours)
# =========================
class ECG200Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _encode_labels_consistently(y_train, y_test):
    all_labels = np.unique(np.concatenate([y_train, y_test], axis=0))
    mapping = {lab: i for i, lab in enumerate(all_labels.tolist())}
    y_train_enc = np.array([mapping[v] for v in y_train], dtype=np.int64)
    y_test_enc  = np.array([mapping[v] for v in y_test], dtype=np.int64)
    return y_train_enc, y_test_enc

def load_ecg200(path="data/ECG200_TRAIN.txt", path_test="data/ECG200_TEST.txt"):
    def load_file(p):
        df = pd.read_csv(p, sep="\s+", engine="python")
        y = df.iloc[:, 0].values
        x = df.iloc[:, 1:].values
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        return x, y

    X_train, y_train = load_file(str(path))
    X_test, y_test = load_file(str(path_test))

    y_train, y_test = _encode_labels_consistently(y_train, y_test)
    return ECG200Dataset(X_train, y_train), ECG200Dataset(X_test, y_test)


class LogisticBasis(nn.Module):
    """
    Differentiable hysteretic logistic basis with up/down branches.

    Output shape matches your original LogisticBasis:
      forward(x): (B, in_dim, num_basis)

    Notes:
    - Uses a persistent buffer prev_x (1, in_dim, num_basis) to infer sweep direction.
    - Uses a *soft* gate g = sigmoid(slope * (x - prev_x)) to select up/down branches,
      which keeps the forward differentiable.
    - prev_x is updated with .detach() (stateful memory, not part of gradient graph).
    """
    def __init__(
        self,
        in_dim: int,
        num_basis: int,
        gate_slope: float = 5.0,   # higher -> more "hard" branch selection
        init_prev: float = 0.0,
        eps: float = 1e-6,
        branch_breaking_point=0.5,
        use_noise=False, noise_std=0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_basis = num_basis
        self.gate_slope = gate_slope
        self.eps = eps
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.branch_breaking_point=branch_breaking_point

        # Learnable parameters for each basis function
        self.k = nn.Parameter(torch.rand(in_dim, num_basis) * 2 + 0.5)  # slope [0.5, 2.5]
        self.Ec = nn.Parameter(torch.rand(in_dim, num_basis) * 2 + 0.5)  # coercive field [0.5, 2.5]
        self.Ps = nn.Parameter(torch.rand(in_dim, num_basis) * 1.5 + 0.5)  # saturation [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(in_dim, num_basis) * 0.1)  # small bias
        self.coef = nn.Parameter(torch.randn(in_dim, num_basis))  # weights
        # Persistent memory (no gradient) for hysteresis
        self.register_buffer("prev_x", torch.zeros(1, in_dim, num_basis))
        self.register_buffer("branch_state", torch.ones(1, in_dim, num_basis))  

    def reset_state(self, value: float = 0.0):
        """Call this at the start of a new sequence/trajectory if you want fresh hysteresis."""
        self.prev_x.zero_()
        self.branch_state.fill_(1.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        if x.dim() != 2 or x.size(1) != self.in_dim:
            raise ValueError(f"x must be (B,{self.in_dim}), got {tuple(x.shape)}")

        B = x.size(0)

        # Expand to (B, in_dim, num_basis)
        x_exp = x.unsqueeze(-1).expand(-1, -1, self.num_basis)

        # Broadcast parameters: (in_dim, num_basis) -> (B, in_dim, num_basis)

        # Two smooth branches (both differentiable)
        # Up branch: centered at +b
        up   = self.Ps * (1 / (1 + torch.exp(-self.k * (x_exp - self.Ec)))) * 2 - self.Ps
        down =  self.Ps * (1 / (1 + torch.exp(-self.k * (x_exp + self.Ec)))) * 2 - self.Ps

        dx = x_exp - self.prev_x

        # Soft gate for gradients to select branches
        g = torch.sigmoid(self.gate_slope * dx)          # (B,in,num_basis)
        self.branch_state = (g > self.branch_breaking_point).float()

        basis = self.branch_state * up + (1.0 - self.branch_state) * down + self.bias

        if self.use_noise:
                noise = torch.randn_like(basis) * self.noise_std
                basis = basis + noise.detach()

        with torch.no_grad():
            self.prev_x.copy_(x_exp[-1:, :, :].detach())

        return basis

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

# ----------- Fully Nonlinear KAN Classifier -----------
class KANClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, num_basis):
        super().__init__()
        self.basis = LogisticBasis(in_dim, num_basis)
        self.activation = nn.Sigmoid()
        self.output = nn.Parameter(torch.randn(in_dim * num_basis, num_classes))

    def forward(self, x):
        phi = self.basis(x)         # (B, in_dim, num_basis)
        phi = self.activation(phi)  # (B, in_dim, num_basis)
        phi_flat = phi.view(x.shape[0], -1)  # (B, in_dim * num_basis)
        return phi_flat @ self.output  # (B, num_classes)

# ----------- Fully Nonlinear KAN RNN Model -----------
class FullyNonlinearKANRNN(nn.Module):
    def __init__(self, input_size=1, seq_len=96, hidden_size=64, num_classes=2, num_basis=10):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.rnn_cell = FullyNonlinearKANCell(input_size, hidden_size, num_basis)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.kan_classifier = KANClassifier(hidden_size, num_classes, num_basis)

    def forward(self, x):  # x: (B, seq_len)
        B = x.size(0)
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(self.seq_len):
            x_t = x[:, t].unsqueeze(1)  # (B, 1)
            h = self.rnn_cell(x_t, h)
        # h = self.norm(h)
        h = self.dropout(h)
        return self.kan_classifier(h)

# =========================
# Neural ODE Model
# =========================
class ODEFunc(nn.Module):
    """
    dz/dt = f(z, t). We ignore t (autonomous), but keep signature for odeint.
    """
    def __init__(self, dim, hidden=128, dropout=0.0, use_ln=True):
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.net = nn.Sequential(
            kan.KAN([dim, hidden]),
            nn.SiLU(),
            nn.Dropout(dropout),
            kan.KAN([hidden, hidden]),
            nn.SiLU(),
            nn.Dropout(dropout),
            kan.KAN([hidden, dim]),
        )
        # Small init helps stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        z = self.ln(z)
        return self.net(z)


class KAN_NODE(nn.Module):
    def __init__(
        self,
        T,
        num_classes=2,
        in_channels=1,
        conv_channels=32,
        ode_hidden=128,
        dropout=0.1,
        solver="dpori5",
        rtol=1e-3,
        atol=1e-4,
    ):
        super().__init__()
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1)
        )

        # no latent projection: state is just pooled conv features (B, C)
        self.to_state = nn.Sequential(
            nn.Flatten(),          # (B, C)
            nn.Dropout(dropout),
        )

        self.odefunc = ODEFunc(conv_channels, hidden=ode_hidden, dropout=dropout, use_ln=True)

        self.head = nn.Sequential(
            nn.LayerNorm(conv_channels),
            nn.Linear(conv_channels, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)                  # (B,1,T)
        z0 = self.to_state(self.stem(x))    # (B,C)

        t = torch.linspace(0.0, 1.0, 9, device=z0.device, dtype=z0.dtype)

        zt = odeint(
            self.odefunc,
            z0,
            t,
            method=self.solver,
        )
        zT = zt[-1]
        return self.head(zT)

# =========================
# Train / Eval
# =========================
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train_KAN_NODE(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=4,
    epochs=100,
    lr=1e-2,
    weight_decay=1e-4,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_ecg200(train_path, test_path)
    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KAN_NODE(
        T=T,
        num_classes=num_classes,
        conv_channels=32,
        ode_hidden=128,
        dropout=0.1,
        solver="rk4",
        rtol=1e-3,
        atol=1e-4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = 0.0
    acc_train_list, acc_test_list = [], []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)

        # Evaluation
        model.eval()
        def evaluate(loader):
            preds, targets = [], []
            for X, y in loader:
                out = model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
            return accuracy_score(targets, preds)
        train_loss = running / len(train_ds)
        acc_train = evaluate(train_loader)
        acc_test = evaluate(test_loader)
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        acc = eval_acc(model, test_loader, "cpu")
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:3d} | train_loss {train_loss:.6f} | test_acc {acc*100:.2f}%")

    return model, acc_train_list, acc_test_list

def train_KAN_RNN(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    epochs=100):
    train_set, test_set = load_ecg200(train_path, test_path)
    # plot_example_signals(train_set)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4)

    model = FullyNonlinearKANRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    acc_train_list, acc_test_list = [], []
    running = 0.0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * y_batch.size(0)
        scheduler.step()

        # Evaluation
        model.eval()
        def evaluate(loader):
            preds, targets = [], []
            for X, y in loader:
                out = model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
            return accuracy_score(targets, preds)
        train_loss = running / len(train_set)
        acc_train = evaluate(train_loader)
        acc_test = evaluate(test_loader)
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        acc = eval_acc(model, test_loader, "cpu")
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.6f} | test_acc {acc*100:.2f}%")
    return model, acc_train_list, acc_test_list



#turns each latent scalar into a small learned function
class KANFeatureMixer(nn.Module):
    """
    Turn x:(B,D) -> phi_flat:(B, D*num_basis), with an extra nonlinearity.
    """
    def __init__(self, dim, num_basis, act=nn.Sigmoid()):
        super().__init__()
        self.basis = LogisticBasis(dim, num_basis)
        self.act = act

    def forward(self, x):
        phi = self.basis(x)           # (B, D, K)
        phi = self.act(phi)           # (B, D, K)
        return phi.reshape(x.size(0), -1)  # (B, D*K)


class MLPKANODEFunc(nn.Module):
    """
    dh/dt = f(h, t) with KANFeatureMixer + KAN blocks, stabilized WITHOUT modifying LogisticBasis.
    """
    def __init__(self, latent_dim=64, num_basis=10, hidden=128,
                 h_bound=1.0, dh_clip=10.0, init_out_std=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_basis = num_basis
        self.h_bound = h_bound
        self.dh_clip = dh_clip

        # normalize + bound state before basis to avoid exp overflow in logistic basis
        self.ln = nn.LayerNorm(latent_dim)

        self.h_feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())  # (B, D*K)

        # KAN blocks (may not have bias/weight exposed)
        self.kan1 = kan.KAN([latent_dim * num_basis, hidden])
        self.act  = nn.SiLU()
        self.kan2 = kan.KAN([hidden, hidden])

        # final projection with standard Linear so we can init small + has bias
        self.out = nn.Linear(hidden, latent_dim, bias=True)

        # small init on final linear => small dh/dt at start
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, mean=0.0, std=init_out_std)

        # Learnable scale on vector field (starts small)
        self.log_alpha = nn.Parameter(torch.tensor(-3.0))  # alpha ≈ softplus(-3) ≈ 0.05

        # Optional additional global scale (keep if you want)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, t, h):
        # 1) normalize + bound h (no LogisticBasis modification needed)
        h = self.ln(h)
        h = self.h_bound * torch.tanh(h / self.h_bound)  # bounds values into ~[-h_bound, h_bound]

        # 2) basis features
        phi = self.h_feat(h)  # (B, D*K)

        phi = torch.nan_to_num(phi, nan=0.0, posinf=1e3, neginf=-1e3)

        # 3) KAN-based transformation
        z = self.kan1(phi)
        z = self.act(z)
        z = self.kan2(z)
        z = self.act(z)

        # 4) output derivative in R^{latent_dim} with a controlled linear head
        dh = self.out(z)

        # 5) scale + clamp
        alpha = F.softplus(self.log_alpha)
        dh = self.scale * alpha * dh

        return dh

class No_MLP_KANODEFunc(nn.Module):
    """
    dh/dt = f(h, t) where f is built using LogisticBasis features (KAN-like).
    """
    def __init__(self, latent_dim=64, num_basis=10, hidden=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_basis = num_basis

        #Before computing the ODE dynamics, expand each latent dimension into a learned set of n
        #onlinear logistic basis functions, then flatten them.
        self.feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())
        # MLP on top of KAN features to produce dh/dt in R^{latent_dim}
        # A small scale can help stability early on
        self.proj = nn.Linear(latent_dim * num_basis, latent_dim)

        # Optional: small init helps stability
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    def forward(self, t, h):
        phi = self.feat(h)          # (B, D*K)
        dh = self.proj(phi)         # (B, D)
        # safety
        assert dh.shape == h.shape, (dh.shape, h.shape)
        return dh
    
# -------------------- KanFet NODE -----------------------

class KanFet_NODE(nn.Module):
    """
    Input x: (B, T). Encode to latent h0, integrate ODE, decode final h(T) -> logits.
    """
    def __init__(
        self,
        T: int,
        num_classes: int,
        latent_dim: int = 64,
        num_basis: int = 10,
        ode_hidden: int = 128,
        dropout: float = 0.1,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Simple encoder: time-series -> latent init
        # (You can swap this with conv1d if you want)
        self.encoder = nn.Linear(T, latent_dim)

        self.odefunc = No_MLP_KANODEFunc(
            latent_dim=latent_dim,
            num_basis=num_basis,
            hidden=ode_hidden,
        )
        self.dropout = nn.Dropout(dropout)

        # KAN-style classifier head (logistic basis on latent)
        self.cls_feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())
        self.cls = nn.Linear(latent_dim * num_basis, num_classes)

    def forward(self, x):  # x: (B, T)
        B = x.size(0)
        h0 = self.encoder(x)  # (B, latent_dim)

        # Integrate from t=0..1; we only need final state.
        # If you want intermediate states, set t_eval to more points.
        t_eval = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        h_traj = odeint(
            self.odefunc,
            h0,
            t_eval,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )  # (2, B, latent_dim)

        hT = h_traj[-1]  # (B, latent_dim)
        hT = self.dropout(hT)

        feat = self.cls_feat(hT)      # (B, latent_dim*num_basis)
        logits = self.cls(feat)       # (B, num_classes)
        return logits


def train_kan_fet_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=4,
    epochs=200,
    lr=1e-2,
    weight_decay=1e-4,
    device=None,
    # model hyperparams
    latent_dim=1,
    num_basis=10,
    ode_hidden=128,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-3,
    atol=1e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # assumes you already have:
    #   train_ds, test_ds = load_ecg200(train_path, test_path)
    train_ds, test_ds = load_ecg200(train_path, test_path)

    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KanFet_NODE(
        T=T,
        num_classes=num_classes,
        latent_dim=latent_dim,
        num_basis=num_basis,
        ode_hidden=ode_hidden,
        dropout=dropout,
        solver=solver,
        rtol=rtol,
        atol=atol,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = 0.0
    train_loss_list, test_acc_list = [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        train_loss_list.append(train_loss)
        acc = eval_acc(model, test_loader, device)


        

        # Evaluation
        model.eval()
        def evaluate(loader):
            preds, targets = [], []
            for X, y in loader:
                out = model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
            return accuracy_score(targets, preds)
        acc_train = evaluate(train_loader)
        acc_test = evaluate(test_loader)
        test_acc_list.append(acc_test)
        acc = eval_acc(model, test_loader, "cpu")
        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"test_acc {acc*100:.2f}% | "
            )
    return model, train_loss_list, test_acc_list


# -------------------- KanFet MLP -----------------------
def integrate_euler(odefunc, h0, t0=0.0, t1=1.0, n_steps=8):
    """Fixed-step explicit Euler. Returns hT."""
    h = h0
    dt = (t1 - t0) / n_steps
    t = torch.as_tensor(t0, device=h0.device, dtype=h0.dtype)
    for _ in range(n_steps):
        dh = odefunc(t, h)
        h = h + dt * dh
        t = t + dt
    return h

def integrate_rk2(odefunc, h0, t0=0.0, t1=1.0, n_steps=8):
    """Fixed-step RK2 (midpoint). Returns hT."""
    h = h0
    dt = (t1 - t0) / n_steps
    t = torch.as_tensor(t0, device=h0.device, dtype=h0.dtype)
    for _ in range(n_steps):
        k1 = odefunc(t, h)
        k2 = odefunc(t + 0.5 * dt, h + 0.5 * dt * k1)
        h = h + dt * k2
        t = t + dt
    return h

def integrate_rk4(odefunc, h0, t0=0.0, t1=1.0, n_steps=8):
    """Fixed-step RK4. Returns hT."""
    h = h0
    dt = (t1 - t0) / n_steps
    t = torch.as_tensor(t0, device=h0.device, dtype=h0.dtype)
    for _ in range(n_steps):
        k1 = odefunc(t, h)
        k2 = odefunc(t + 0.5 * dt, h + 0.5 * dt * k1)
        k3 = odefunc(t + 0.5 * dt, h + 0.5 * dt * k2)
        k4 = odefunc(t + dt,       h + dt * k3)
        h = h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
    return h

class KanFet_MLP_Euler_Rollout(nn.Module):
    """
    Same as your model, but replaces odeint with fixed-step rollout.
    """
    def __init__(
        self,
        T: int,
        num_classes: int,
        latent_dim: int = 64,
        num_basis: int = 10,
        ode_hidden: int = 128,
        dropout: float = 0.1,
        integrator: str = "euler",   # "euler" | "rk2" | "rk4"
        n_steps: int = 8,          # fixed number of steps from 0..1
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.integrator = integrator
        self.n_steps = n_steps

        self.encoder = nn.Linear(T, latent_dim)

        self.odefunc = MLPKANODEFunc(
            latent_dim=latent_dim,
            num_basis=num_basis,
            hidden=ode_hidden,
        )

        self.dropout = nn.Dropout(dropout)
        self.cls_feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())
        self.cls = nn.Linear(latent_dim * num_basis, num_classes)

    def _integrate(self, h0):
        if self.integrator == "euler":
            return integrate_euler(self.odefunc, h0, 0.0, 1.0, self.n_steps)
        elif self.integrator == "rk2":
            return integrate_rk2(self.odefunc, h0, 0.0, 1.0, self.n_steps)
        elif self.integrator == "rk4":
            return integrate_rk4(self.odefunc, h0, 0.0, 1.0, self.n_steps)
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def forward(self, x):  # x: (B, T)
        h0 = self.encoder(x)     # (B, latent_dim)
        hT = self._integrate(h0) # (B, latent_dim)
        hT = self.dropout(hT)

        feat = self.cls_feat(hT)
        logits = self.cls(feat)
        return logits


def train_kan_fet_mlp(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=4,
    epochs=200,
    lr=1e-2,
    weight_decay=1e-4,
    device=None,
    latent_dim=64,
    num_basis=10,
    ode_hidden=128,
    dropout=0.1,
    integrator="euler",
    n_steps=8,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_ecg200(train_path, test_path)
    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KanFet_MLP_Euler_Rollout(
        T=T,
        num_classes=num_classes,
        latent_dim=latent_dim,
        num_basis=num_basis,
        ode_hidden=ode_hidden,
        dropout=dropout,
        integrator=integrator,
        n_steps=n_steps,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = 0.0
    train_loss_list, test_acc_list = [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        acc = eval_acc(model, test_loader, device)

        best = max(best, acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(acc)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"test_acc {acc*100:.2f}% | "
                f"best {best*100:.2f}%"
            )

    return model, train_loss_list, test_acc_list

# -------------------- KanFet MLP NODE No Latent Embedding --------------------

class KanFet_MLP_NODE_NOLatentEmbeddings(nn.Module):
    """
    Input x: (B, T). Encode to latent h0, integrate ODE, decode final h(T) -> logits.
    """
    def __init__(
        self,
        T: int,
        num_classes: int,
        latent_dim: int = 64,
        num_basis: int = 10,
        ode_hidden: int = 128,
        dropout: float = 0.1,
        solver: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Simple encoder: time-series -> latent init
        # (You can swap this with conv1d if you want)
        self.encoder = nn.Linear(T, latent_dim)
   

        self.odefunc =  MLPKANODEFunc(
            latent_dim=latent_dim,
            num_basis=num_basis,
            hidden=ode_hidden,
        )

        self.dropout = nn.Dropout(dropout)

        # KAN-style classifier head (logistic basis on latent)
        self.cls_feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())
        self.cls = nn.Linear(latent_dim * num_basis, num_classes)

    def forward(self, x):  # x: (B, T)
        B = x.size(0)
        h0 = self.encoder(x)  # (B, latent_dim)

        # Integrate from t=0..1; we only need final state.
        # If you want intermediate states, set t_eval to more points.
        t_eval = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)

        h_traj = odeint(
            self.odefunc,
            h0,
            t_eval,
            method=self.solver,
        )  # (2, B, latent_dim)

        hT = h_traj[-1]  # (B, latent_dim)
        hT = self.dropout(hT)

        feat = self.cls_feat(hT)      # (B, latent_dim*num_basis)
        logits = self.cls(feat)       # (B, num_classes)
        return logits



def train_kan_fet_mlp_node_nolatentembeddings(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=4,
    epochs=200,
    lr=1e-2,
    weight_decay=1e-4,
    device=None,
    # model hyperparams
    latent_dim=64,
    num_basis=10,
    ode_hidden=128,
    dropout=0.1,
    solver="rk4",
    rtol=1e-3,
    atol=1e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # assumes you already have:
    #   train_ds, test_ds = load_ecg200(train_path, test_path)
    train_ds, test_ds = load_ecg200(train_path, test_path)

    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KanFet_MLP_NODE_NOLatentEmbeddings(
        T=T,
        num_classes=num_classes,
        latent_dim=latent_dim,
        num_basis=num_basis,
        ode_hidden=ode_hidden,
        dropout=dropout,
        solver=solver,
        rtol=rtol,
        atol=atol,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = 0.0
    train_loss_list, test_acc_list = [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        acc = eval_acc(model, test_loader, device)
        train_loss_list.append(train_loss)
        test_acc_list.append(acc)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"test_acc {acc*100:.2f}% | "
            )

    return model, train_loss_list, test_acc_list


# -------------------- KanFet MLP NODE Latent Space Embedded --------------------


class KanFet_MLP_NODE(nn.Module):
    """
    Input x: (B, T). Encode to latent h0, integrate ODE, decode final h(T) -> logits.
    """
    def __init__(
        self,
        T: int,
        num_classes: int,
        latent_dim: int = 64,
        num_basis: int = 10,
        ode_hidden: int = 128,
        dropout: float = 0.1,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Simple encoder: time-series -> latent init
        # (You can swap this with conv1d if you want)
        self.encoder = nn.Linear(T, latent_dim)


        self.odefunc =  MLPKANODEFunc(
            latent_dim=latent_dim,
            num_basis=num_basis,
            hidden=ode_hidden,
        )

        self.dropout = nn.Dropout(dropout)

        # KAN-style classifier head (logistic basis on latent)
        self.cls_feat = KANFeatureMixer(latent_dim, num_basis, act=nn.Sigmoid())
        self.cls = nn.Linear(latent_dim * num_basis, num_classes)

    def forward(self, x):  # x: (B, T)
        B = x.size(0)
        h0 = self.encoder(x)  # (B, latent_dim)

        # Integrate from t=0..1; we only need final state.
        # If you want intermediate states, set t_eval to more points.
        #t_eval = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        t_eval = torch.linspace(0.0, 1.0, 9, device=x.device, dtype=torch.float64)
        h_traj = odeint(
            self.odefunc,
            h0,
            t_eval,
            method="dopri5",
            rtol=self.rtol,
            atol=self.atol,
        )  # (2, B, latent_dim)

        hT = h_traj[-1]  # (B, latent_dim)
        hT = self.dropout(hT)

        feat = self.cls_feat(hT)      # (B, latent_dim*num_basis)
        logits = self.cls(feat)       # (B, num_classes)
        return logits


def train_kan_fet_mlp_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=4,
    epochs=200,
    lr=1e-2,
    weight_decay=1e-4,
    device=None,
    # model hyperparams
    latent_dim=64,
    num_basis=10,
    ode_hidden=128,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-3,
    atol=1e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # assumes you already have:
    #   train_ds, test_ds = load_ecg200(train_path, test_path)
    train_ds, test_ds = load_ecg200(train_path, test_path)

    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KanFet_MLP_NODE(
        T=T,
        num_classes=num_classes,
        latent_dim=latent_dim,
        num_basis=num_basis,
        ode_hidden=ode_hidden,
        dropout=dropout,
        solver=solver,
        rtol=rtol,
        atol=atol,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = 0.0
    train_loss_list, test_acc_list = [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        train_loss_list.append(train_loss)
        #acc = eval_acc(model, test_loader, device)


        # Evaluation
        model.eval()
        def evaluate(loader):
            preds, targets = [], []
            for X, y in loader:
                out = model(X)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
            return accuracy_score(targets, preds)
        acc_train = evaluate(train_loader)
        acc_test = evaluate(test_loader)
        test_acc_list.append(acc_test)
        acc = eval_acc(model, test_loader, "cpu")
        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"test_acc {acc*100:.2f}% | "
            )
    return model, train_loss_list, test_acc_list

# =========================
# Train / Eval
# =========================
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)





if __name__ == "__main__":
    # Put ECG200_TRAIN.txt / ECG200_TEST.txt in the same folder or pass paths.

    EPOCHS = 100

    print("Training KAN-NODE..hello.")
    '''
    node_model, node_model_train_acc, node_model_test_acc = train_KAN_NODE(
        train_path="data/ECG200_TRAIN.txt",
        test_pah="data/ECG200_TEST.txt",
        batch_size=4,
        epochs=EPOCHS,
        lr=1e-2,
    )

    print("Training KanFet-RNN...")
    base_knn_model, base_knn_train_acc, base_knn_test_acc = train_KAN_RNN(epochs=EPOCHS)

    '''
    print("Training KanFEPA-NODE...")
    kan_fet_ode_model, kan_fet_ode_train_acc, kan_fet_ode_test_acc = train_kan_fet_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-4,
    device="cpu",

    # Neural ODE / KAN knobs
    latent_dim=1,
    num_basis=12,
    ode_hidden=128,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-2,
    atol=1e-3,
   )


    '''

    print("Training KanFet-MLP...")
    kan_fet_mlp_model, kan_fet_mlp_train_acc, kan_fet_mlp_test_acc = train_kan_fet_mlp (
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-4,
    device="cpu",

    # Neural ODE / KAN knobs
    latent_dim=1,
    num_basis=12,
    ode_hidden=128,
    dropout=0.1)

    
    print("Training KanFet-MLP-NODE (No Latent Embeddings)...")
    kanfet_mlp_nolatent_model, kanfet_mlp_nolatent_train_acc, kanfet_mlp_nolatent_test_acc = \
        train_kan_fet_mlp_node_nolatentembeddings (
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-4,
    device="cpu",

    # Neural ODE / KAN knobs
    latent_dim=1,
    num_basis=12,
    ode_hidden=128,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-2,
    atol=1e-3,
   )
    '''

    print("Training KanFEPA-MLP-NODE (Latent Space Embedded)...")

    kan_fet_mlp_node_model, kan_fet_mlp_node_train_acc, kan_fet_mlp_node_test_acc = train_kan_fet_mlp_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-4,
    device="cpu",

    # Neural ODE / KAN knobs
    latent_dim=64,
    num_basis=12,
    ode_hidden=128,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-2,
    atol=1e-3,
   )

    plt.figure(figsize=(10, 7))
    KANFET_RNN_COLOR  = "#1f77b4"  # blue
    KAN_NODE_COLOR  = "#d62728"  # red
    KAN_FET_ODE_color = "#2ca02c"  # green

    # --- additional colors --
    KANFET_MLP_COLOR        = "#ff7f0e"  # orange
    KANFET_MLP_NODE_COLOR = "#9467bd" # purple
    KANFET_MLP_NODE_NOLATENTEMBEDDINGS_COLOR   = "#8c564b"  # brown

    plt.plot(base_knn_train_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN (No Latent Space) Train Loss", linewidth=3)
    plt.plot(kan_fet_ode_train_acc, color=KAN_FET_ODE_color, label="KanFEPA-NODE (Latent Space Embedded) Train Loss", linewidth=3)
    #plt.plot(node_model_train_acc, color=KAN_NODE_COLOR, label="KAN-NODE (No Latent Space) Train Loss", linewidth=3)
    #plt.plot(kan_fet_mlp_train_acc, color=KANFET_MLP_COLOR, label="KanFEPA-MLP (No Latent Space) Train Loss", linewidth=3)
    #plt.plot(kanfet_mlp_nolatent_train_acc, color=KANFET_MLP_NODE_NOLATENTEMBEDDINGS_COLOR, label="KanFEPA-MLP-NODE (No Latent Space) Train Loss", linewidth=3)
    plt.plot(kan_fet_mlp_node_train_acc, color=KANFET_MLP_NODE_COLOR, label="KanFEPA-MLP-NODE (Latent Space Embedded) Train Loss", linewidth=3)


    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Training Loss", fontsize=30)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 7))
    plt.plot(base_knn_test_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN (No Latent Space) Test Accuracy", linewidth=3)
    plt.plot(kan_fet_ode_test_acc, color=KAN_FET_ODE_color, label="KanFEPA-NODE (Latent Space Embedded) Test Accuracy", linewidth=3)
    #plt.plot(node_model_test_acc, color=KAN_NODE_COLOR, label="KAN-NODE (No Latent Space) Test Accuracy", linewidth=3)
    #plt.plot(kan_fet_mlp_test_acc, color=KANFET_MLP_COLOR, label="KanFEPA-MLP (No Latent Space) Test Accuracy", linewidth=3)
    #plt.plot(kanfet_mlp_nolatent_test_acc, color=KANFET_MLP_NODE_NOLATENTEMBEDDINGS_COLOR, label="KanFEPA-MLP-NODE (No Latent Space) Test Accuracy", linewidth=3)
    plt.plot(kan_fet_mlp_node_test_acc, color=KANFET_MLP_NODE_COLOR, label="KanFEPA-MLP-NODE (Latent Space Embedded) Test Accuracy", linewidth=3)

    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Test Accuracy", fontsize=30)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
