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
from kan_diffusion import kan
from efficient_kan.efficientkan import KANFET
from ferro_class import FerroelectricBasis

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
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, num_classes=2,
        dropout=0.1, bidirectional=False):
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


   

class FullyNonlinearKANCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_basis):
        super(FullyNonlinearKANCell, self).__init__()
        self.input_basis =  FerroelectricBasis(input_size, hidden_size, num_basis)
        self.hidden_basis = FerroelectricBasis(hidden_size, hidden_size, num_basis)

        self.activation = nn.Sigmoid()

        self.hidden_size = hidden_size
        self.num_basis = num_basis

    def forward(self, x_t, h_prev):  # x_t: (B, input_size), h_prev: (B, hidden_size)
        outs = []
        if h_prev is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        B = x_t.size(0)
        for b in range(B):
            xb = x_t[b:b+1]      # (1, input_size)
            hb = h_prev[b:b+1]   # (1, hidden_size)
            x_phi = self.input_basis(xb).view(1, -1)
            h_phi = self.hidden_basis(hb).view(1, -1)
            combined = torch.cat([x_phi, h_phi], dim=1)
            out = self.activation(combined)[:, :self.hidden_size]
            outs.append(out)
        return torch.cat(outs, dim=0)  # (B, hidden_size)

# ----------- Fully Nonlinear KAN Classifier -----------
class KANClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_basis):
        super().__init__()
        self.basis =  FerroelectricBasis(in_dim, hidden_dim, num_basis)
        self.activation = nn.Sigmoid()
        self.head = nn.Linear(hidden_dim, num_classes)  # <— FIX
        # explicit linear head (no nn.Linear module equivalence below)
        #self.W = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.02)
        #self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):  # x: (B, in_dim)
        B = x.size(0)
        logits_list = []
        for b in range(B):
            xb = x[b:b+1]                   # (1, in_dim)   batch=1 for basis
            phi = self.activation(self.basis(xb))  # (1, hidden_dim)
            logits = self.head(phi)   #phi @ self.W + self.b         (1, num_classes)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)   

# ----------- Fully Nonlinear KAN RNN Model -----------
class FullyNonlinearKANRNN(nn.Module):
    def __init__(self, input_size=1, seq_len=96, hidden_size=64, num_classes=2, num_basis=10):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.rnn_cell = FullyNonlinearKANCell(input_size, hidden_size, num_basis)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.kan_classifier = KANClassifier(hidden_size, hidden_size, num_classes, num_basis)

    def forward(self, x):  # x: (B, seq_len) # x: (B, T)
        B = x.size(0)
        hs = []

        for b in range(B):
            xb = x[b:b+1]  # (1, T)
            h = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)

            for t in range(self.seq_len):
                x_t = xb[:, t].unsqueeze(1)  # (1, 1)
                h = self.rnn_cell(x_t, h)    # basis sees batch=1

                if h is None:
                    raise RuntimeError(f"rnn_cell returned None at t={t} (sample b={b})")

            h = self.norm(h)      # (1, hidden)
            h = self.dropout(h)
            hs.append(h)

        h = torch.cat(hs, dim=0)  # (B, hidden)
        return self.kan_classifier(h) 




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
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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
# -------------------- RNN-NODE Function --------------------
class KANRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_basis):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = FullyNonlinearKANCell(input_size, hidden_size, num_basis)

    def forward(self, x):  # x: (B, T) or (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, _ = x.shape
        hs = []

        # IMPORTANT: basis can only handle batch=1, so loop over batch dimension
        for b in range(B):
            xb = x[b:b+1]  # (1, T, D)
            h = torch.zeros(1, self.hidden_size, device=x.device, dtype=x.dtype)
            for t in range(T):
                h = self.cell(xb[:, t, :], h)  # each call sees B=1
            hs.append(h)
        return torch.cat(hs, dim=0)  # (B, hidden_size)

class KANAutonomousODEFunc(nn.Module):
    """
    dh/dt = f(h) using ONLY FerroelectricBasis + elementwise params.
    No nn.Linear.
    """
    def __init__(self, hidden_size, num_basis):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis

        self.basis = FerroelectricBasis(hidden_size, hidden_size, num_basis)

        # elementwise gains + bias (diagonal affine), not a dense Linear
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.act = nn.Tanh()

    def forward(self, t, h):  # t required by odeint 
        phi = self.basis(h)

        # If phi is already (B, hidden_size), great.
        # If phi is (B, hidden_size, num_basis) or (B, something), reduce to (B, hidden_size).
        if phi.dim() == 3:
            # common: (B, hidden, K) -> reduce over K
            phi = phi.mean(dim=-1)
        elif phi.dim() == 2:
            pass
        else:
            # last-resort: flatten then reshape (only if you know shapes!)
            phi = phi.view(h.size(0), self.hidden_size, -1).mean(dim=-1)

        dh = self.act(phi) * self.gain + self.bias
        return dh

class NODE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_basis, solver="rk4", rtol=1e-3, atol=1e-4, ode_T=10.0, n_eval=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.ode_T = ode_T
        self.n_eval = n_eval

        # Lift x_t -> h0 (this can be linear or basis; use linear first for stability)
        self.lift = FerroelectricBasis(input_size, hidden_size, num_basis)

        # Autonomous ODE in hidden space
        self.odefunc = KANAutonomousODEFunc(hidden_size, num_basis)

    def forward(self, xb):  # xb: (1, T, D)
        t = torch.linspace(0.0, float(self.ode_T), steps=1)

        T = xb.size(1)
        zs = []
        for tt in range(T):
            x_t = xb[:, tt, :]                  # (1, D)
            h0 = self.lift(x_t)                 # (1, H)
            hT = odeint(self.odefunc, h0, t,
                        method=self.solver, rtol=self.rtol, atol=self.atol)[-1]  # (1, H)
            zs.append(hT)
        return torch.cat(zs, dim=0).unsqueeze(0)   # (1, T, H)



class NODE_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_classes=2, num_basis=10,
                 odefunc=None,
                 solver="rk4", rtol=1e-3, atol=1e-4, ode_T=10.0, n_eval=5,  dropout=0.1):
        super().__init__()

        self.ode_seq = NODE_Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_basis=num_basis,
            solver=solver, rtol=rtol, atol=atol, ode_T=ode_T, n_eval=n_eval
        )
        self.rnn = FullyNonlinearKANCell(input_size, hidden_size, num_basis)
        self.dropout = nn.Dropout(dropout)
        self.head =  KANClassifier(hidden_size, hidden_size, num_classes, num_basis) 

    def forward(self, x):  # x: (B,T) or (B,T,D)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, _, _ = x.shape

        feats = []
        for b in range(B):
            xb = x[b:b+1]                 # (1,T,D)
            z_seq = self.ode_seq(xb)      # (1,T,H)

            h = torch.zeros(1, self.ode_seq.hidden_size, device=x.device, dtype=x.dtype)  # (1,H)
            for t in range(z_seq.size(1)):
                z_t = z_seq[:, t, :]      # (1,H)
                h = self.rnn(z_t, h)      # (1,H)
            h = self.dropout(h)           # (1,H)
            feats.append(h)               # append ONCE per sample
        feats = torch.cat(feats, dim=0)   # (B,H)
        logits = self.head(feats)         # (B,C)
        return logits   # (B,C)



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
        ode_T=ode_T,
        solver=solver,
        rtol=rtol,
        atol=atol,
        n_eval=n_eval,
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

            # IMPORTANT: always forward with full batch_size
            #reset_stateful_ferro_buffers(model)
            optimizer.zero_grad()  
            logits = model(x)      # slice back to real batch
            loss = criterion(logits, y)   # use true labels only
            loss.backward()
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

# -------------------- KanFet MLP NODE Latent Space Embedded --------------------

# -------------------- MLP-NODE Function --------------------

class KANFetODEFunc(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_basis: int, h_bound: float = 1.0):
        super().__init__()
        self.h_bound = h_bound #specifying bound is super important to prevent dx underflow during solving
        #we should use SmoothFerroElectricBasis below; 
        #otherwise, if we use KANFerroelectricBasis, solver states will explode 
        #due to non-differentiable branch switching
        self.fc1 = FerroelectricBasis(latent_dim, hidden_dim, num_basis)
        self.act = nn.Tanh()  # or Sigmoid, GELU, etc.
        self.fc2 =  FerroelectricBasis(hidden_dim, latent_dim, num_basis)
        self.MLP = nn.Linear(hidden_dim, latent_dim, bias=True)
        


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
    '''
    print("Training Baseline Digital RNN...")
    digital_rnn_model, digital_rnn_train_loss, digital_rnn_test_acc = train_rnn_baseline(
        train_path="data/ECG200_TRAIN.txt",
        test_path="data/ECG200_TEST.txt",
        batch_size=16,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        clip_grad=1.0,
    )


    print("Training KanFEPA-RNN...")
    base_rnn_model, base_rnn_train_acc, base_rnn_test_acc = train_KAN_RNN(epochs=EPOCHS)


    '''
    print("Training KanFEPA-RNN-NODE (Latent Space Embedded)...")

    
    rnn_node_model, rnn_node_train_acc, rnn_node_test_acc = train_rnn_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=8,
    epochs=EPOCHS,
    lr=5e-3,
    weight_decay=1e-4,
    device="cpu",

    # Neural ODE / KAN knobs
    hidden_size=64,
    num_basis=12,
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
    Digital_RNN_COLOR = "#2ca02c"  # green
    KANFET_RNN_COLOR  = "#1f77b4"  # blue
    KAN_NODE_COLOR  = "#d62728"  # red
    KAN_FET_ODE_color = "#2ca02c"  # green

    # --- additional colors --
    KANFET_MLP_COLOR        = "#ff7f0e"  # orange
    KANFET_MLP_NODE_COLOR = "#9467bd" # purple
    KANFET_MLP_NODE_NOLATENTEMBEDDINGS_COLOR   = "#8c564b"  # brown

    plt.plot(digital_rnn_train_loss, color=Digital_RNN_COLOR, label="Digital RNN (Baseline) Train Loss", inewidth=3)
    plt.plot(base_rnn_train_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN (No Latent Space) Train Loss", linewidth=3)
    plt.plot(rnn_node_train_acc, color=KAN_FET_ODE_color, label="KanFEPA-RNN-NODE (Latent Space Embedded) Train Loss", linewidth=3)
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
    plt.plot(base_rnn_test_acc, color=KANFET_RNN_COLOR, label="KanFEPA-RNN (No Latent Space) Test Accuracy", linewidth=3)
    plt.plot(rnn_node_test_acc, color=KAN_FET_ODE_color, label="KanFEPA-RNN-NODE (Latent Space Embedded) Test Accuracy", linewidth=3)
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
    '''