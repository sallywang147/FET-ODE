import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# pip install torchdiffeq
from torchdiffeq import odeint
from ferro_class import NoisyFerroelectricBasis, NoisyBatchedFerroelectricBasis


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

def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)



# -----------------------------
# Digital RNN Model
# -----------------------------
class Digital_RNN(nn.Module):
    """
    A conventional sequence classifier:
      x: (B, T)  -> (B, T, 1)
      RNN/GRU/LSTM -> last hidden -> linear head -> logits
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=30, num_classes=2,
        dropout=0.0, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            nonlinearity="tanh",
            )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        # x: (B, T) or (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)

        _, h = self.rnn(x)
        h_n = h
        last = h_n[-self.num_directions:]  # (num_directions, B, H)
        # concat directions
        last = last.transpose(0, 1).contiguous().view(x.size(0), -1)  # (B, H*num_directions)

        last = self.dropout(last)
        logits = self.head(last)  # (B, C)
        return logits



# -----------------------------
# Train / Eval
# -----------------------------


def train_rnn_baseline(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=16,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=64,
    num_layers=1,
    dropout=0.1,
    bidirectional=False,
    clip_grad=1.0,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = load_ecg200(train_path, test_path)
    T = train_ds.X.shape[1]
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Digital_RNN(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    print(f"Trainable parameters: {count_trainable_params(model),}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    running = 0.0
    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            #reset_stateful_ferro_buffers(model)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    train_loss_list, test_acc_list = [], []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            # IMPORTANT: always forward with full batch_size
            #reset_stateful_ferro_buffers(model)
            optimizer.zero_grad()  
            logits = model(x)      # slice back to real batch
            loss = criterion(logits, y)   # use true labels only
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)
        train_loss_list.append(train_loss)
        acc_train = evaluate(train_loader)
        acc_test  = evaluate(test_loader)
        test_acc_list.append(acc_test)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"train_acc {acc_train*100:.2f}% | "
                f"test_acc {acc_test*100:.2f}% | "
            )

    return model, train_loss_list, test_acc_list


# -------------------- FEPA RNN--------------------

class FullyNonlinearKANCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_basis):
        super().__init__()
        self.input_basis = NoisyBatchedFerroelectricBasis(input_size, hidden_size, num_basis)
        self.hidden_basis = NoisyBatchedFerroelectricBasis(hidden_size, hidden_size, num_basis)
        self.activation = torch.tanh
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
    def __init__(self, in_dim,  hidden_dim, num_classes, num_basis):
        super().__init__()
        self.basis = NoisyBatchedFerroelectricBasis(in_dim,  hidden_dim, num_basis)
        self.activation = torch.tanh
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim


    def forward(self, x):
        output = nn.Parameter(torch.randn(self.hidden_dim, self.num_classes))
        phi = self.basis(x)         # (B, in_dim, num_basis)
        phi = self.activation(phi)  # (B, in_dim, num_basis)
        phi_flat = phi.view(x.shape[0], -1)  # (B, in_dim * num_basis)
        return phi_flat @ output  # (B, num_classes)

# ----------- Fully Nonlinear KAN RNN Model -----------
class FullyNonlinearKANRNN(nn.Module):
    def __init__(self, input_size=1, seq_len=96, hidden_size=64, num_classes=2, num_basis=10):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.rnn_cell = FullyNonlinearKANCell(input_size, hidden_size, num_basis)
        #self.norm = nn.LayerNorm(hidden_size)
        #self.dropout = nn.Dropout(0.1)
        self.kan_classifier = KANClassifier(hidden_size, hidden_size, num_classes, num_basis)

    def forward(self, x):  # x: (B, seq_len)
        B = x.size(0)
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(self.seq_len):
            x_t = x[:, t].unsqueeze(1)  # (B, 1)
            h = self.rnn_cell(x_t, h)
        # h = self.norm(h)
        #h = self.dropout(h)
        return self.kan_classifier(h)


@torch.no_grad()
def reset_stateful_ferro_buffers(m):
    for mm in m.modules():
        if mm.__class__.__name__ == "NoisyFerroelectricBasis":
            if hasattr(mm, "prev_x") and mm.prev_x is not None:
                mm.prev_x.zero_()
            if hasattr(mm, "branch_sign") and mm.branch_sign is not None:
                mm.branch_sign.fill_(1.0)


def train_KAN_RNN(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    epochs=100):
    train_set, test_set = load_ecg200(train_path, test_path)
    # plot_example_signals(train_set)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4)

    model = FullyNonlinearKANRNN()
    print(f"Trainable parameters: {count_trainable_params(model),}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            reset_stateful_ferro_buffers(model)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    train_loss_list, test_acc_list = [], []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            # IMPORTANT: always forward with full batch_size
            reset_stateful_ferro_buffers(model)
            optimizer.zero_grad()  
            logits = model(x)      # slice back to real batch
            loss = criterion(logits, y)   # use true labels only
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()
        train_loss = running / len(train_set)
        train_loss_list.append(train_loss)
        acc_train = evaluate(train_loader)
        acc_test  = evaluate(test_loader)
        test_acc_list.append(acc_test)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"train_acc {acc_train*100:.2f}% | "
                f"test_acc {acc_test*100:.2f}% | "
            )

    return model, train_loss_list, test_acc_list

#-----------------------RNN-NODE-----------------------------


class LinearInterp1D:
    """
    Cheap batched linear interpolation for x(t) along a fixed time grid.
    Expects:
      - t_grid: (T,) increasing
      - x_grid: (T, D)   (per-sample)
    Returns x(t): (1, D) for scalar t
    """
    def __init__(self, t_grid, x_grid):
        self.t = t_grid
        self.x = x_grid
        self.T = t_grid.numel()

    def __call__(self, t_query):
        # clamp to grid range
        t = torch.clamp(t_query, self.t[0], self.t[-1])

        # find right index i s.t. t[i-1] <= t < t[i]
        # searchsorted returns in [0..T]
        i = torch.searchsorted(self.t, t).clamp(1, self.T - 1)
        t0, t1 = self.t[i - 1], self.t[i]
        x0, x1 = self.x[i - 1], self.x[i]

        w = (t - t0) / (t1 - t0 + 1e-12)
        return (1.0 - w) * x0 + w * x1  # (D,)


class InputDrivenKANODEFunc(nn.Module):
    """
    Non-autonomous ODE:
      dh/dt = f(h, x(t))
    Uses FerroelectricBasis blocks + diagonal gain/bias (no dense Linear required).
    """
    def __init__(self, input_size, hidden_size, num_basis):
        super().__init__()
        self.hidden_size = hidden_size

        # map concat([h, x]) -> hidden
        self.basis = NoisyFerroelectricBasis(hidden_size + input_size, hidden_size, num_basis)

        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.act = nn.Tanh()

        # filled per-sample before calling odeint
        self._interp = None

    def set_interpolator(self, interp_callable):
        self._interp = interp_callable

    def forward(self, t, h):
        # h: (1, H) or (B, H). Here we run per-sample because your basis has state.
        assert self._interp is not None, "Interpolator not set. Call set_interpolator() before odeint."
        x_t = self._interp(t).unsqueeze(0)  # (1, D)

        hx = torch.cat([h, x_t], dim=-1)    # (1, H + D)
        phi = self.basis(hx)                # (1, H)  (your basis returns (B, out_dim))

        dh = self.act(phi) * self.gain + self.bias
        return dh


class OneODEEncoder(nn.Module):
    """
    Single ODE solve for the entire sequence.
    Produces one feature vector per sample (default: final hidden state).
    """
    def __init__(self, input_size, hidden_size, num_basis,
                 solver="rk4", rtol=1e-3, atol=1e-4):
        super().__init__()
        self.hidden_size = hidden_size
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Lift x0 -> h0 (you can swap this for nn.Linear for stability)
        #If we lift/encode by Basis class, acc gets stuck at 63.64%, no matter how we tune hyperparams 
        self.lift = nn.Linear(input_size, hidden_size) #FerroelectricBasis(input_size, hidden_size, num_basis)

        self.odefunc = InputDrivenKANODEFunc(input_size, hidden_size, num_basis)

    def forward(self, xb):
        """
        xb: (1, T, D)  (per-sample, because your FerroelectricBasis has internal buffers)
        returns: (1, H)
        """
        assert xb.dim() == 3 and xb.size(0) == 1
        T = xb.size(1)
        D = xb.size(2)

        # time grid: normalize to [0, 1]
        t_grid = torch.linspace(0.0, 1.0, steps=T, device=xb.device, dtype=xb.dtype)  # (T,)
        x_grid = xb[0]  # (T, D)

        interp = LinearInterp1D(t_grid, x_grid)
        self.odefunc.set_interpolator(interp)

        # initial state from first sample
        h0 = self.lift(x_grid[0:1])  # (1, H)

        # single ODE solve; we only need the final state for classification
        h_traj = odeint(self.odefunc, h0, t_grid,
                        method=self.solver, rtol=self.rtol, atol=self.atol)  # (T, 1, H)
        hT = h_traj[-1]  # (1, H)
        return hT


class NODE_RNN(nn.Module):
    """
    Full model: one ODE encoder + head.
    """
    def __init__(self, input_size=1, hidden_size=64, num_classes=2, num_basis=10,
                 solver="rk4", rtol=1e-3, atol=1e-4, dropout=0.1):
        super().__init__()
        self.enc = OneODEEncoder(input_size, hidden_size, num_basis, solver=solver, rtol=rtol, atol=atol)
        #self.dropout = nn.Dropout(dropout)
        self.rnn_cell = FullyNonlinearKANCell(hidden_size, hidden_size, num_basis)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B,T) or (B,T,D)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, D = x.shape

        feats = []
        for b in range(B):
            xb = x[b:b+1]                 # (1, T, D)

            # IMPORTANT: call encoder once
            z_seq = self.enc(xb)          # expected (1, T, H)

            # make sure z_seq is (T, H) for indexing by time
            if z_seq.dim() == 3:
                z_seq = z_seq[0]          # (T, H)
            elif z_seq.dim() != 2:
                raise RuntimeError(f"Unexpected z_seq shape: {tuple(z_seq.shape)}")

            h = torch.zeros(1, self.enc.hidden_size, device=x.device, dtype=x.dtype)  # (1, H)

            for t in range(z_seq.size(0)):
                z_t = z_seq[t:t+1]        # (1, H)  <-- safest slice (never becomes 1D)
                h = self.rnn_cell(z_t, h)
            feats.append(h)   # (B, C)
        feats = torch.cat(feats, dim=0)   # (B, H)
        logits = self.head(feats)         # (B, C)
        return logits




# ----------------------------
# Training loop adapted for RNN-NODE
# ----------------------------

def train_rnn_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    hidden_size=64,
    num_basis=10,
    dropout=0.1,
    ode_T=1.0,
    solver="rk4",
    rtol=1e-3,
    atol=1e-4,
    n_eval=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = load_ecg200(train_path, test_path)
    num_classes = int(torch.max(train_ds.y).item() + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    model = NODE_RNN(
        input_size=1,
        hidden_size=hidden_size,
        num_basis=num_basis,
        num_classes=num_classes,
        dropout=dropout,
        solver=solver,
        rtol=rtol,
        atol=atol,
    ).to(device)


    print(f"Trainable parameters: {count_trainable_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            #reset_stateful_ferro_buffers(model)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    train_loss_list, test_acc_list = [], []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            torch.autograd.set_detect_anomaly(True)

            # IMPORTANT: always forward with full batch_size
            #reset_stateful_ferro_buffers(model)
            optimizer.zero_grad()  
            logits = model(x)      # slice back to real batch
            loss = criterion(logits, y)   # use true labels only
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y.size(0)


        train_loss = running / len(train_ds)
        train_loss_list.append(train_loss)

        acc_train = evaluate(train_loader)
        acc_test  = evaluate(test_loader)
        test_acc_list.append(acc_test)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"train_acc {acc_train*100:.2f}% | "
                f"test_acc {acc_test*100:.2f}% | "
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



class KANFetODEFunc(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_basis: int, h_bound: float = 1.0):
        super().__init__()
        self.h_bound = h_bound #specifying bound is super important to prevent dx underflow during solving
        #we should use SmoothFerroElectricBasis below; 
        #otherwise, if we use KANFerroelectricBasis, solver states will explode 
        #due to non-differentiable branch switching
        self.fc1 = NoisyFerroelectricBasis(latent_dim, hidden_dim, num_basis)
        self.act = nn.Tanh()  # or Sigmoid, GELU, etc.
        self.fc2 =  NoisyFerroelectricBasis(hidden_dim, latent_dim, num_basis)

    def forward(self, t, h):
        # h: (B, latent_dim) (or (latent_dim,) if B==1 in some uses)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        #line below is very important,otherwise underflow during ODE solving occurs
        h = self.h_bound * torch.tanh(h / self.h_bound) 
        z = self.fc1(h)      # (B, hidden_dim)
        z = self.act(z)
        dh = self.fc2(z)     # (B, latent_dim)

        # Hard safety: prevent NaN/Inf from killing the solver
        dh = torch.nan_to_num(dh, nan=0.0, posinf=1e3, neginf=-1e3)

        # Optional: bound slope to reduce stiffness
        dh = torch.clamp(dh, -50.0, 50.0)
        
        return dh


#code below has underflow error; requiring fix 
class KanFet_MLP_NODE(nn.Module):
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
       
        self.encoder = nn.Linear(T, latent_dim)
        self.odefunc = KANFetODEFunc(latent_dim=latent_dim, hidden_dim=ode_hidden, num_basis=num_basis)
        self.dropout = nn.Dropout(dropout)
        self.cls =  nn.Linear(latent_dim, num_classes)

    def forward(self, x):  # x: (B, T)
        h0 = self.encoder(x)  # (B, latent_dim)
        B = x.size(0)
        device, dtype = x.device, x.dtype
        t = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        logits_list = []
        for b in range(B):
            xb = x[b:b+1]                     # (1, T)
            h0 = self.encoder(xb)             # (1, latent_dim)

            # ODE solve with batch=1 so FerroelectricBasis doesn't crash
            hT = odeint(self.odefunc, h0, t,
                        method=self.solver, rtol=self.rtol, atol=self.atol)[-1]  # (1, latent_dim)

            hT = self.dropout(hT)
        return self.cls(hT)

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
    solver="euler",
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

    print(f"Trainable parameters: {count_trainable_params(model),}")
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





if __name__ == "__main__":
    # Put ECG200_TRAIN.txt / ECG200_TEST.txt in the same folder or pass paths.

    EPOCHS = 100

    mlp_node_model, mlp_node_train_acc, mlp_node_test_acc =  train_kan_fet_mlp_node(
        train_path="data/ECG200_TRAIN.txt",
        test_path="data/ECG200_TEST.txt",
        batch_size=1,
        epochs=EPOCHS,
        lr=1e-5,
        weight_decay=1e-4,
        device=None,
        # model hyperparams
        latent_dim=64,
        num_basis=12,
        ode_hidden=128,
        dropout=0.1,
        solver="euler",
        rtol=1e-3,
        atol=1e-4,
    )
   
    print("Training Baseline Digital RNN...")
    digital_rnn_model, digital_rnn_train_loss, digital_rnn_test_acc = train_rnn_baseline(
        train_path="data/ECG200_TRAIN.txt",
        test_path="data/ECG200_TEST.txt",
        batch_size=8,
        epochs=EPOCHS,
        lr=1e-5,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        bidirectional=True,
        clip_grad=1.0,
    )


    print("Training KanFEPA-RNN...")
    base_rnn_model, base_rnn_train_acc, base_rnn_test_acc = train_KAN_RNN(epochs=EPOCHS)



    print("Training KanFEPA-RNN-NODE (Latent Space Embedded)...")

    
    rnn_node_model, rnn_node_train_acc, rnn_node_test_acc = train_rnn_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=1,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-5,
    device="cpu",

    # Neural ODE / KAN knobs
    hidden_size=64,
    num_basis=12,
    dropout=0.1,
    solver="dopri5",
    rtol=1e-2,
    atol=1e-3,
   )
   

    plt.figure(figsize=(10, 7))
    Digital_RNN_COLOR = "#2ca02c"  # green
    KANFET_RNN_COLOR  = "#1f77b4"  # blue
    KAN_NODE_COLOR  = "#d62728"  # red
    MLP_NODE_COLOR = "#bcbd22"  # yellow / olive

    plt.plot(digital_rnn_train_loss, color=Digital_RNN_COLOR, label="Digital RNN (Baseline) Train Loss", linewidth=3)
    plt.plot(base_rnn_train_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN Train Loss", linewidth=3)
    plt.plot(rnn_node_train_acc, color=KAN_NODE_COLOR, label="KanFEPA-RNN-NODE Train Loss", linewidth=3)
    plt.plot(mlp_node_train_acc, color=MLP_NODE_COLOR, label="KanFEPA-MLP-NODE Train Loss", linewidth=3)


    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Training Loss (20% Noise Per Basis)", fontsize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 7))
    plt.plot(digital_rnn_test_acc, color=Digital_RNN_COLOR, label="Digital RNN (Baseline) Test Accuracy", linewidth=3)
    plt.plot(base_rnn_test_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN Test Accuracy", linewidth=3)
    plt.plot(rnn_node_test_acc, color=KAN_NODE_COLOR, label="KanFEPA-RNN-NODE Test Accuracy", linewidth=3)
    plt.plot(mlp_node_test_acc, color=MLP_NODE_COLOR, label="KanFEPA-MLP-NODE Test Accuracy", linewidth=3)


    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Test Accuracy (20% Noise Per Basis)", fontsize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
