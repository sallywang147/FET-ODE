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


def ferro_params(layer, i_idx, o_idx, b_idx):
    w  = float(layer.coef[i_idx, o_idx, b_idx].detach().cpu())
    Ps = float(layer.Ps [i_idx, o_idx, b_idx].detach().cpu())
    Ec = float(layer.Ec [i_idx, o_idx, b_idx].detach().cpu())
    k  = float(layer.k  [i_idx, o_idx, b_idx].detach().cpu())
    return w, Ps, Ec, k



def visualize_clean_vs_noisy_ferro_layer(
    layer,
    layer_name="layer",
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),
    noise_std=0.2,      # override layer.noise_std if provided
    seed=0,              # seed for noisy curve so it’s repeatable
):
    """
    Overlay clean vs noisy hysteresis loops for a single FerroelectricBasis-like layer.

    Works for BOTH:
      - BatchedFerroelectricBasis
      - FerroelectricBasis

    Assumes:
      - layer.forward(x, return_activations=True) returns (_, basis, _)
      - basis shape: (B, in_dim, out_dim, num_basis)
      - layer has .use_noise and .noise_std
    """
    device = next(layer.parameters()).device

    i_idx, o_idx = which_in_out
    i_idx = int(np.clip(i_idx, 0, layer.in_dim - 1))
    o_idx = int(np.clip(o_idx, 0, layer.out_dim - 1))

    nb = min(max_basis, layer.num_basis)

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    n_cols = min(2, nb)
    n_rows = (nb + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if nb == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # save & maybe override noise_std
    old_use = layer.use_noise
    old_std = layer.noise_std
    if noise_std is not None:
        layer.noise_std = float(noise_std)

    def run_sweep(use_noise, seed_for_noise=None):
        layer.use_noise = use_noise
        if hasattr(layer, "reset_state"):
            layer.reset_state()

        if seed_for_noise is not None:
            # Make the noise deterministic for visualization
            torch.manual_seed(seed_for_noise)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_for_noise)

        P_up = [[] for _ in range(nb)]
        P_dn = [[] for _ in range(nb)]

        # Up sweep
        for e in E_up:
            x_in = torch.full((1, layer.in_dim), float(e.item()),
                              device=device, dtype=torch.float32)
            with torch.no_grad():
                _, basis, _ = layer(x_in, return_activations=True)
            for b in range(nb):
                P_up[b].append(basis[0, i_idx, o_idx, b].item())

        # Down sweep
        for e in E_dn:
            x_in = torch.full((1, layer.in_dim), float(e.item()),
                              device=device, dtype=torch.float32)
            with torch.no_grad():
                _, basis, _ = layer(x_in, return_activations=True)
            for b in range(nb):
                P_dn[b].append(basis[0, i_idx, o_idx, b].item())

        return P_up, P_dn

    # ---- run clean then noisy (with reset between) ----
    clean_up, clean_dn = run_sweep(use_noise=False, seed_for_noise=None)
    noisy_up, noisy_dn = run_sweep(use_noise=True,  seed_for_noise=seed)

    # ---- plot ----
    for b_idx in range(nb):
        ax = axes[b_idx]
        w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)

        ax.plot(E_up.detach().cpu().numpy(), np.array(clean_up[b_idx]), linewidth=2, label="Clean up")
        ax.plot(E_dn.detach().cpu().numpy(), np.array(clean_dn[b_idx]), linewidth=2, label="Clean down")

        ax.plot(E_up.detach().cpu().numpy(), np.array(noisy_up[b_idx]),  linestyle="--", linewidth=2,
                label=f"Noisy up (std={layer.noise_std:.3f})")
        ax.plot(E_dn.detach().cpu().numpy(), np.array(noisy_dn[b_idx]),  linestyle="--", linewidth=2,
                label="Noisy down")

        ax.set_xlabel("Electric Field (E)")
        ax.set_ylabel("Polarization (P)")
        ax.set_title(
            f"{layer_name} | basis {b_idx}\n"
            f"w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # hide unused
    for j in range(nb, len(axes)):
        axes[j].set_visible(False)

    # restore
    layer.use_noise = old_use
    layer.noise_std = old_std

    fig.suptitle(f"Clean vs Noisy Hysteresis: {layer_name}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()

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
        self.input_basis = FerroelectricBasis(input_size, hidden_size, num_basis)
        self.hidden_basis = FerroelectricBasis(hidden_size, hidden_size, num_basis)
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
        self.basis = FerroelectricBasis(in_dim,  hidden_dim, num_basis)
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


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def ferro_params(layer, i_idx, o_idx, b_idx):
    """
    Works for BOTH BatchedFerroelectricBasis and FerroelectricBasis as you defined them:
      k, Ec, Ps, coef all have shape (in_dim, out_dim, num_basis)
    """
    w  = float(layer.coef[i_idx, o_idx, b_idx].detach().cpu())
    Ps = float(layer.Ps [i_idx, o_idx, b_idx].detach().cpu())
    Ec = float(layer.Ec [i_idx, o_idx, b_idx].detach().cpu())
    k  = float(layer.k  [i_idx, o_idx, b_idx].detach().cpu())
    return w, Ps, Ec, k


def visualize_noisy_ferro_layer(
    layer,
    layer_name="layer",
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),
    noise_std=0.2,     # override layer.noise_std if provided
    seed=0,             # seed for deterministic noise
):
    """
    Plot ONLY the noisy hysteresis curves for a single layer (use_noise=True).

    Works for BOTH:
      - BatchedFerroelectricBasis
      - FerroelectricBasis

    Assumes:
      - layer.forward(x, return_activations=True) returns (_, basis, _)
      - basis shape: (B, in_dim, out_dim, num_basis)
      - layer has .use_noise and .noise_std
    """
    device = next(layer.parameters()).device

    i_idx, o_idx = which_in_out
    i_idx = int(np.clip(i_idx, 0, layer.in_dim - 1))
    o_idx = int(np.clip(o_idx, 0, layer.out_dim - 1))

    nb = min(max_basis, layer.num_basis)

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    n_cols = min(2, nb)
    n_rows = (nb + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if nb == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Save and force noisy mode
    old_use = getattr(layer, "use_noise", False)
    old_std = getattr(layer, "noise_std", None)

    if hasattr(layer, "use_noise"):
        layer.use_noise = True
    if noise_std is not None and hasattr(layer, "noise_std"):
        layer.noise_std = float(noise_std)

    # Deterministic noise for visualization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reset once for proper up->down hysteresis sweep
    if hasattr(layer, "reset_state"):
        layer.reset_state()

    def _make_input(e_scalar):
        return torch.full((1, layer.in_dim), float(e_scalar), dtype=torch.float32, device=device)

    # Precompute all basis curves (so titles can be set cleanly)
    P_up = [[] for _ in range(nb)]
    P_dn = [[] for _ in range(nb)]

    # Up sweep
    for e in E_up:
        x_in = _make_input(e.item())
        with torch.no_grad():
            _, basis, _ = layer(x_in, return_activations=True)
        for b in range(nb):
            P_up[b].append(basis[0, i_idx, o_idx, b].item())

    # Down sweep
    for e in E_dn:
        x_in = _make_input(e.item())
        with torch.no_grad():
            _, basis, _ = layer(x_in, return_activations=True)
        for b in range(nb):
            P_dn[b].append(basis[0, i_idx, o_idx, b].item())

    x_up = E_up.detach().cpu().numpy()
    x_dn = E_dn.detach().cpu().numpy()

    for b_idx in range(nb):
        ax = axes[b_idx]
        w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)

        ax.plot(x_up, np.array(P_up[b_idx]), linewidth=2, label=f"Noisy up (std={layer.noise_std:.3f})")
        ax.plot(x_dn, np.array(P_dn[b_idx]), linewidth=2, label="Noisy down")

        ax.set_xlabel("Electric Field (E)")
        ax.set_ylabel("Polarization (P)")
        ax.set_title(
            f"{layer_name} | basis {b_idx}\n"
            f"w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused
    for j in range(nb, len(axes)):
        axes[j].set_visible(False)

    # Restore
    if hasattr(layer, "use_noise"):
        layer.use_noise = old_use
    if hasattr(layer, "noise_std"):
        layer.noise_std = old_std

    fig.suptitle(f"Noisy Hysteresis: {layer_name}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()


def visualize_FEPA_RNN_hysteresis_noisy(
    model,
    epoch,
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),
    share_x=True,
    noise_std=0.2,   # override model layer noise_std if provided
    seed=0,           # deterministic noise for visualization
    save_dir="Noisy_FEPA_RNN_Hysteresis",
):
    """
    Visualize NOISY hysteresis loops for ALL ferroelectric bases in the KanFEPA_RNN model:
      1) model.rnn_cell.input_basis
      2) model.rnn_cell.hidden_basis
      3) model.kan_classifier.basis

    This version forces use_noise=True while plotting.
    """
    model.eval()
    device = next(model.parameters()).device

    bases = [
        ("rnn_cell.input_basis",   model.rnn_cell.input_basis),
        ("rnn_cell.hidden_basis",  model.rnn_cell.hidden_basis),
        ("kan_classifier.basis",   model.kan_classifier.basis),
    ]

    # Field sweeps
    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    in_dim_idx, out_dim_idx = which_in_out

    # One clean grid for all rows
    num_basis_to_show = min(max_basis, min(layer.num_basis for _, layer in bases))
    n_rows, n_cols = len(bases), num_basis_to_show

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    # Deterministic noise for visualization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def _make_input(layer, e_scalar):
        return torch.full((1, layer.in_dim), float(e_scalar), dtype=torch.float32, device=device)

    for row, (name, layer) in enumerate(bases):
        i_idx = int(np.clip(in_dim_idx, 0, layer.in_dim - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer.out_dim - 1))

        # Save and force noisy mode for this layer
        old_use = getattr(layer, "use_noise", False)
        old_std = getattr(layer, "noise_std", None)

        if hasattr(layer, "use_noise"):
            layer.use_noise = True
        if noise_std is not None and hasattr(layer, "noise_std"):
            layer.noise_std = float(noise_std)

        # Reset once per layer so up->down shows hysteresis
        if hasattr(layer, "reset_state"):
            layer.reset_state()

        for b_idx in range(num_basis_to_show):
            ax = axes[row, b_idx]
            P_up, P_dn = [], []

            # Up sweep
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    _, basis, _ = layer(x_in, return_activations=True)
                P_up.append(basis[0, i_idx, o_idx, b_idx].item())

            # Down sweep
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    _, basis, _ = layer(x_in, return_activations=True)
                P_dn.append(basis[0, i_idx, o_idx, b_idx].item())

            w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2,
                    label=f"Noisy up (std={layer.noise_std:.3f})")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Noisy down")

            ax.set_title(
                f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}",
                fontsize=10
            )
            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization (P)")
            ax.grid(True, alpha=0.3)
            if b_idx == 0:
                ax.legend(fontsize=9)

        # Restore layer noise settings
        if hasattr(layer, "use_noise"):
            layer.use_noise = old_use
        if hasattr(layer, "noise_std"):
            layer.noise_std = old_std

    fig.suptitle(
        f"20% Noise Added: KanFEPA-RNN Ferroelectric Hysteresis Loops",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"KanFEPA_RNN_hysteresis_noisy_epoch{epoch}.png")
    plt.savefig(out_path, dpi=200)
    #plt.show()

    return out_path


def visualize_FEPA_RNN_hysteresis(
    model,
    epoch,
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),   # (in_dim_idx, out_dim_idx) for reading basis[0, i, o, b]
    share_x=True,
):
    """
    Visualize hysteresis loops for ALL ferroelectric bases in your model:
      1) model.rnn_cell.input_basis
      2) model.rnn_cell.hidden_basis
      3) model.kan_classifier.basis   (classifier basis)

    Layout:
      rows = 3 (input_basis, hidden_basis, classifier_basis)
      cols = num_basis_to_show (up to max_basis, shared across rows)

    Assumes BatchedFerroelectricBasis supports:
      - .in_dim, .out_dim, .num_basis
      - .reset_state()
      - forward(x, return_activations=True) -> (phi, basis, coef)
        where basis is indexed like basis[0, in_dim_idx, out_dim_idx, basis_idx]
    """
    model.eval()
    device = next(model.parameters()).device

    bases = [
        ("rnn_cell.input_basis",   model.rnn_cell.input_basis),
        ("rnn_cell.hidden_basis",  model.rnn_cell.hidden_basis),
        ("kan_classifier.basis",   model.kan_classifier.basis),
    ]

    # field sweeps
    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    in_dim_idx, out_dim_idx = which_in_out

    # Choose a single #basis to show across all rows for a clean grid
    num_basis_to_show = min(max_basis, min(layer.num_basis for _, layer in bases))
    n_rows, n_cols = len(bases), num_basis_to_show

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    def _make_input(layer, e_scalar):
        # layer expects (B, in_dim)
        return torch.full((1, layer.in_dim), float(e_scalar), dtype=torch.float32, device=device)

    for row, (name, layer) in enumerate(bases):
        # Clamp indices to the layer's ranges
        i_idx = int(np.clip(in_dim_idx, 0, layer.in_dim - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer.out_dim - 1))

        # IMPORTANT: reset ONCE per layer so up->down shows hysteresis
        if hasattr(layer, "reset_state"):
            layer.reset_state()

        for b_idx in range(num_basis_to_show):
            ax = axes[row, b_idx]
            P_up, P_dn = [], []

            # Up sweep
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    _, basis, coef = layer(x_in, return_activations=True)
                P_up.append(basis[0, i_idx, o_idx, b_idx].item())

            # Down sweep
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    _, basis, coef = layer(x_in, return_activations=True)
                P_dn.append(basis[0, i_idx, o_idx, b_idx].item())

            # Optional: show a per-basis coefficient if your layer provides it
            weight_str = ""
            try:
                w = float(coef[0, i_idx, b_idx].item())
                weight_str = f", w={w:.3f}"
            except Exception:
                pass

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2, label="Up")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Down")

            w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)
            ax.set_title(f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}", fontsize=10)
            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization (P)")
            ax.grid(True, alpha=0.3)
            if b_idx == 0:
                ax.legend(fontsize=9)

    fig.suptitle(
        "KanFEPA-RNN Ferroelectric Hysteresis Loops for RNN and Classifier",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f"KanFEPA_RNN_Hysteresis/KanFEPA_RNN_hysteresis_epoch{epoch}.png")
    #plt.show()

@torch.no_grad()
def reset_stateful_ferro_buffers(m):
    for mm in m.modules():
        if mm.__class__.__name__ == "FerroelectricBasis":
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
    '''
    visualize_clean_vs_noisy_ferro_layer(
    model.rnn_cell.input_basis,
    layer_name="rnn_cell.input_basis",
    noise_std=0.2,
    seed=0
    )

    visualize_clean_vs_noisy_ferro_layer(
        model.rnn_cell.hidden_basis,
        layer_name="rnn_cell.hidden_basis",
        noise_std=0.2,
        seed=0
    )
    '''

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
            #remove the _noisy in function sig to visualize the clean version
            visualize_FEPA_RNN_hysteresis_noisy(
                model,
                ep, 
                field_range=(-5, 5),
                n_points=200,
                max_basis=5,
                which_in_out=(0, 0),
            )
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
        self.basis = FerroelectricBasis(hidden_size + input_size, hidden_size, num_basis)

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
    def __init__(self, input_size, hidden_size, num_basis,
                 solver="rk4", rtol=1e-3, atol=1e-4, return_traj=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.return_traj = return_traj

        self.lift = nn.Linear(input_size, hidden_size)
        self.odefunc = InputDrivenKANODEFunc(input_size, hidden_size, num_basis)

    def forward(self, xb, return_traj=None):
        assert xb.dim() == 3 and xb.size(0) == 1
        return_traj = self.return_traj if return_traj is None else return_traj

        T = xb.size(1)
        t_grid = torch.linspace(0.0, 1.0, steps=T, device=xb.device, dtype=xb.dtype)
        x_grid = xb[0]  # (T, D)

        interp = LinearInterp1D(t_grid, x_grid)
        self.odefunc.set_interpolator(interp)

        # IMPORTANT: reset hysteresis state before solve (if your basis is stateful)
        if hasattr(self.odefunc.basis, "reset_state"):
            self.odefunc.basis.reset_state()

        h0 = self.lift(x_grid[0:1])  # (1, H)

        h_traj = odeint(self.odefunc, h0, t_grid,
                        method=self.solver, rtol=self.rtol, atol=self.atol)  # (T, 1, H)

        if return_traj:
            return h_traj  # (T, 1, H)
        else:
            return h_traj[-1]  # (1, H)


class NODE_RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_classes=2, num_basis=10,
                 solver="rk4", rtol=1e-3, atol=1e-4):
        super().__init__()
        self.enc = OneODEEncoder(input_size, hidden_size, num_basis, solver=solver, rtol=rtol, atol=atol)
        self.rnn_cell = FullyNonlinearKANCell(hidden_size, hidden_size, num_basis)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        feats = []
        for b in range(B):
            xb = x[b:b+1]  # (1, T, D)

            # get full trajectory
            h_traj = self.enc(xb, return_traj=True)   # (T, 1, H)
            z_seq = h_traj[:, 0, :]                   # (T, H)

            # reset RNN hysteresis per sample (VERY important if stateful)
            if hasattr(self.rnn_cell.input_basis, "reset_state"):
                self.rnn_cell.input_basis.reset_state()
            if hasattr(self.rnn_cell.hidden_basis, "reset_state"):
                self.rnn_cell.hidden_basis.reset_state()

            h = torch.zeros(1, self.enc.hidden_size, device=device, dtype=dtype)

            for t in range(z_seq.size(0)):
                z_t = z_seq[t:t+1]  # (1, H)
                h = self.rnn_cell(z_t, h)

            feats.append(h)

        feats = torch.cat(feats, dim=0)   # (B, H)
        return self.head(feats)
'''
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
        self.head = nn.Linear(hidden_size, num_classes) #digital classifier

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
'''

def visualize_all_ferroelectric_bases_NODE_RNN(
    model,
    epoch, 
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),     # (in_dim_idx, out_dim_idx) if basis returns (B,in_dim,out_dim,num_basis)
    style="grid",            # "grid" or "separate"
    share_x=True,
):
    """
    Visualize hysteresis loops for ALL FerroelectricBasis / BatchedFerroelectricBasis modules inside NODE_RNN:

      A) model.enc.odefunc.basis                 : FerroelectricBasis(hidden+input -> hidden)
      B) model.rnn_cell.input_basis              : BatchedFerroelectricBasis(hidden -> hidden)  (input to RNN cell is z_t)
      C) model.rnn_cell.hidden_basis             : BatchedFerroelectricBasis(hidden -> hidden)

    Notes:
    - Your earlier visualizer assumed `layer(x, return_activations=True) -> (_, basis, coef)`
      and `basis` indexable as basis[0, i, o, b].
    - Here we support BOTH common cases:
        1) BatchedFerroelectricBasis: forward(x, return_activations=True) -> (_, basis, coef)
           with basis shaped (B, in_dim, out_dim, num_basis)
        2) FerroelectricBasis: may either:
           (a) support return_activations=True similarly, OR
           (b) only return output phi (B, out_dim) (in which case we plot output dim vs E, not per-basis internals)

    If your FerroelectricBasis DOES support return_activations, you will get true per-basis loops.
    Otherwise, you’ll still get a useful “effective output hysteresis” curve.

    Usage:
        visualize_all_ferroelectric_bases_in_NODE_RNN(model, which_in_out=(0,0), max_basis=5)
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- Collect all base modules we want to visualize ----
    bases = [
        ("enc.odefunc.basis",        model.enc.odefunc.basis),
        ("rnn_cell.input_basis",     model.rnn_cell.input_basis),
        ("rnn_cell.hidden_basis",    model.rnn_cell.hidden_basis),
        # If you later add classifier basis, append it here too.
        # ("classifier.basis",       model.some_classifier.basis),
    ]

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)
    in_dim_idx, out_dim_idx = which_in_out

    # Determine number of basis functions to show for each layer (if available)
    def _num_basis(layer):
        return getattr(layer, "num_basis", None)

    # Use a single column count for a clean grid; if a layer lacks num_basis, we still show 1 col for it.
    nb_list = [(_num_basis(layer) or 1) for _, layer in bases]
    num_basis_to_show = min(max_basis, min(nb_list))

    n_rows = len(bases)
    n_cols = num_basis_to_show if style == "grid" else 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    # Normalize axes indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    # ---- Helpers ----
    def _reset(layer):
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    def _make_input(layer, e_scalar):
        # Most of your basis layers take (B, in_dim)
        in_dim = getattr(layer, "in_dim", None)
        if in_dim is None:
            # fallback: guess 1
            in_dim = 1
        return torch.full((1, in_dim), float(e_scalar), dtype=torch.float32, device=device)

    def _call_with_activations(layer, x_in):
        """
        Try calling layer(x, return_activations=True).
        Return a tuple (mode, basis_tensor, coef_tensor, phi_tensor)
          - mode == "basis": basis_tensor is not None (B,in_dim,out_dim,num_basis)
          - mode == "phi": only phi_tensor is valid (B,out_dim)
        """
        # Best-effort: some implementations accept return_activations keyword
        try:
            out = layer(x_in, return_activations=True)
            # Expect (_, basis, coef) or (phi, basis, coef)
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                phi, basis, coef = out[0], out[1], out[2]
                # basis should be tensor with 4 dims
                if torch.is_tensor(basis) and basis.dim() == 4:
                    return "basis", basis, coef, phi
                # otherwise fall back to plotting phi
                return "phi", None, None, phi
            # If returned only phi
            if torch.is_tensor(out):
                return "phi", None, None, out
            return "phi", None, None, out[0] if isinstance(out, (tuple, list)) else None
        except TypeError:
            # Doesn't accept return_activations -> just call normally
            phi = layer(x_in)
            return "phi", None, None, phi

    # ---- Main plotting loop ----
    for row, (name, layer) in enumerate(bases):
        _reset(layer)

        # clamp indices for basis-tensor reading
        layer_in = getattr(layer, "in_dim", 1)
        layer_out = getattr(layer, "out_dim", getattr(layer, "hidden_size", 1))
        i_idx = int(np.clip(in_dim_idx, 0, layer_in - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer_out - 1))

        # If the layer has num_basis, we plot per basis. Otherwise, we plot only output o_idx.
        layer_nb = getattr(layer, "num_basis", None)
        plot_cols = num_basis_to_show if (style == "grid" and layer_nb is not None) else 1

        for col in range(plot_cols):
            b_idx = col  # basis index (if applicable)
            ax = axes[row, col] if style == "grid" else axes[row, 0]

            P_up, P_dn = [], []
            weight_str = ""

            # ---- up ----
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    mode, basis, coef, phi = _call_with_activations(layer, x_in)

                if mode == "basis":
                    P_up.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    # fallback: plot "effective output" at output dim o_idx
                    # phi shape expected (1, out_dim)
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_up.append(phi[0, o_idx].item())
                    else:
                        P_up.append(phi.reshape(-1)[0].item())

            # ---- down ----
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    mode, basis, coef, phi = _call_with_activations(layer, x_in)

                if mode == "basis":
                    P_dn.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_dn.append(phi[0, o_idx].item())
                    else:
                        P_dn.append(phi.reshape(-1)[0].item())

            # Try to get coefficient if available
            if layer_nb is not None:
                try:
                    # common: coef[0, i_idx, b_idx]
                    w = float(coef[0, i_idx, b_idx].item())
                    weight_str = f", w={w:.3f}"
                except Exception:
                    weight_str = ""

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2, label="Up")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Down")

            # Title
            if layer_nb is None:
                ax.set_title(f"{name} (effective output))", fontsize=5)
            else:
                ax.set_title(f"{name} | basis {b_idx}{weight_str}", fontsize=10)
                w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)
                ax.set_title(f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}", fontsize=10)

            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization / Response")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=9)

        # Hide unused columns for layers that don't have num_basis (grid mode)
        if style == "grid" and layer_nb is None:
            for col in range(1, n_cols):
                axes[row, col].set_visible(False)

    fig.suptitle(
        "KanFEPA_RNN_NODE Ferroelectric Hysteresis Loops for RNN and ODE",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f"KanFEPA_RNN_NODE_Hysteresis/kanFEPA_RNN_NODE_epoch{epoch}.png")
    plt.tight_layout()
    #plt.show()


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def ferro_params(layer, i_idx, o_idx, b_idx):
    # Works for BOTH BatchedFerroelectricBasis and FerroelectricBasis as you defined them
    w  = float(layer.coef[i_idx, o_idx, b_idx].detach().cpu())
    Ps = float(layer.Ps [i_idx, o_idx, b_idx].detach().cpu())
    Ec = float(layer.Ec [i_idx, o_idx, b_idx].detach().cpu())
    k  = float(layer.k  [i_idx, o_idx, b_idx].detach().cpu())
    return w, Ps, Ec, k


def visualize_all_ferroelectric_bases_NODE_RNN_noisy(
    model,
    epoch,
    field_range=(-5, 5),
    n_points=200,
    max_basis=5,
    which_in_out=(0, 0),
    style="grid",
    share_x=True,
    noise_std=0.2,   # override per-layer noise_std for visualization
    seed=0,           # deterministic noise
    save_dir="Noisy_KanFEPA_RNN_NODE_Hysteresis",
):
    """
    NOISY version of visualize_all_ferroelectric_bases_NODE_RNN.

    Forces use_noise=True for each FerroelectricBasis / BatchedFerroelectricBasis layer while plotting.
    Also fixes coefficient indexing: w is read from layer.coef[i,o,b] (not from returned `coef`).

    Layers visualized:
      A) model.enc.odefunc.basis
      B) model.rnn_cell.input_basis
      C) model.rnn_cell.hidden_basis

    If a layer doesn't support return_activations=True, we fall back to plotting "effective output"
    (phi[:, o_idx]) vs E, but we can only show w/Ps/Ec/k if the layer has those Parameters.
    """
    model.eval()
    device = next(model.parameters()).device

    bases = [
        ("enc.odefunc.basis",        model.enc.odefunc.basis),
        ("rnn_cell.input_basis",     model.rnn_cell.input_basis),
        ("rnn_cell.hidden_basis",    model.rnn_cell.hidden_basis),
    ]

    # deterministic noise for visualization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)
    in_dim_idx, out_dim_idx = which_in_out

    def _num_basis(layer):
        return getattr(layer, "num_basis", None)

    nb_list = [(_num_basis(layer) or 1) for _, layer in bases]
    num_basis_to_show = min(max_basis, min(nb_list))

    n_rows = len(bases)
    n_cols = num_basis_to_show if style == "grid" else 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    def _reset(layer):
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    def _make_input(layer, e_scalar):
        in_dim = getattr(layer, "in_dim", 1)
        return torch.full((1, in_dim), float(e_scalar), dtype=torch.float32, device=device)

    def _call_with_activations(layer, x_in):
        """
        Return (mode, basis, phi):
          - mode=="basis": basis is (B,in_dim,out_dim,num_basis)
          - mode=="phi"  : phi is (B,out_dim) or (B,*) and basis is None
        """
        try:
            out = layer(x_in, return_activations=True)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                # (output, basis, coef) OR (phi, basis, coef)
                phi = out[0]
                basis = out[1] if len(out) > 1 else None
                if torch.is_tensor(basis) and basis.dim() == 4:
                    return "basis", basis, phi
                return "phi", None, phi
            if torch.is_tensor(out):
                return "phi", None, out
            return "phi", None, out[0] if isinstance(out, (tuple, list)) else None
        except TypeError:
            phi = layer(x_in)
            return "phi", None, phi

    # ---- Main plotting loop ----
    for row, (name, layer) in enumerate(bases):
        # clamp indices
        layer_in = getattr(layer, "in_dim", 1)
        layer_out = getattr(layer, "out_dim", getattr(layer, "hidden_size", 1))
        i_idx = int(np.clip(in_dim_idx, 0, layer_in - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer_out - 1))

        # Force noisy mode for this layer
        old_use = getattr(layer, "use_noise", False)
        old_std = getattr(layer, "noise_std", None)

        if hasattr(layer, "use_noise"):
            layer.use_noise = True
        if noise_std is not None and hasattr(layer, "noise_std"):
            layer.noise_std = float(noise_std)

        _reset(layer)

        layer_nb = getattr(layer, "num_basis", None)
        plot_cols = num_basis_to_show if (style == "grid" and layer_nb is not None) else 1

        for col in range(plot_cols):
            b_idx = col
            ax = axes[row, col] if style == "grid" else axes[row, 0]

            P_up, P_dn = [], []

            # Up sweep
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    mode, basis, phi = _call_with_activations(layer, x_in)
                if mode == "basis":
                    P_up.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    # effective output
                    if phi is not None and phi.dim() == 2 and phi.size(1) > o_idx:
                        P_up.append(phi[0, o_idx].item())
                    else:
                        P_up.append(float(phi.reshape(-1)[0].item()) if phi is not None else 0.0)

            # Down sweep
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    mode, basis, phi = _call_with_activations(layer, x_in)
                if mode == "basis":
                    P_dn.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    if phi is not None and phi.dim() == 2 and phi.size(1) > o_idx:
                        P_dn.append(phi[0, o_idx].item())
                    else:
                        P_dn.append(float(phi.reshape(-1)[0].item()) if phi is not None else 0.0)

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2,
                    label=f"Noisy up (std={getattr(layer, 'noise_std', 0.0):.3f})")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Noisy down")

            # Title text
            if layer_nb is None or plot_cols == 1 and layer_nb is None:
                ax.set_title(f"{name} (effective output)", fontsize=10)
            else:
                # show physical params if available
                if all(hasattr(layer, attr) for attr in ("coef", "Ps", "Ec", "k")):
                    w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)
                    ax.set_title(
                        f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}",
                        fontsize=10
                    )
                else:
                    ax.set_title(f"{name}\nBasis b={b_idx}", fontsize=10)

            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization / Response")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=9)

        # hide unused columns if needed
        if style == "grid" and layer_nb is None:
            for col in range(1, n_cols):
                axes[row, col].set_visible(False)

        # restore noise settings for this layer
        if hasattr(layer, "use_noise"):
            layer.use_noise = old_use
        if hasattr(layer, "noise_std"):
            layer.noise_std = old_std

    fig.suptitle(
        f"20% Noise Added:KanFEPA-RNN-NODE NOISY Ferroelectric Hysteresis Loops",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"kanFEPA_RNN_NODE_noisy_epoch{epoch}.png")
    plt.savefig(out_path, dpi=200)
    #plt.show()

    return out_path

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
            '''
            visualize_all_ferroelectric_bases_NODE_RNN(
                model,
                ep, 
                field_range=(-5, 5),
                n_points=200,
                max_basis=5,
                which_in_out=(0, 0),
                style="grid",
            )
            '''
            _ = visualize_all_ferroelectric_bases_NODE_RNN_noisy(
                model, epoch=ep, noise_std=0.2, seed=0
            )

            print(
                f"Epoch {ep:3d} | "
                f"train_loss {train_loss:.6f} | "
                f"train_acc {acc_train*100:.2f}% | "
                f"test_acc {acc_test*100:.2f}% | "
            )

    return model, train_loss_list, test_acc_list
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
       
        self.encoder = nn.Linear(T, latent_dim) #linear layer encoding in digital 
        self.odefunc = KANFetODEFunc(latent_dim=latent_dim, hidden_dim=ode_hidden, num_basis=num_basis)
        self.dropout = nn.Dropout(dropout)
        self.cls =  nn.Linear(latent_dim, num_classes) #MLP classifier in digital 

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


def visualize_all_ferro_bases_KanFEPA_MLP_NODE(
    model,
    epoch,
    field_range=(-5, 5),
    n_points=200,
    max_basis=6,
    which_in_out=(0, 0),     # used only if your FerroelectricBasis exposes basis tensor (B,in_dim,out_dim,num_basis)
    mode="auto",             # "auto" | "basis" | "output"
    share_x=True,
):
    model.eval()
    device = next(model.parameters()).device

    layers = [
        ("odefunc.fc1", model.odefunc.fc1),
        ("odefunc.fc2", model.odefunc.fc2),
    ]

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    in_dim_idx, out_dim_idx = which_in_out

    def _reset(layer):
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    def _make_input(layer, e_scalar):
        in_dim = getattr(layer, "in_dim", 1)
        # feed constant field across all input dims
        return torch.full((1, in_dim), float(e_scalar), dtype=torch.float32, device=device)

    def _try_call(layer, x_in):
        """
        Returns (kind, phi, basis, coef)
          kind: "basis" if we got per-basis tensor; otherwise "output"
        """
        # user can force
        if mode == "output":
            phi = layer(x_in)
            return "output", phi, None, None
        if mode in ("auto", "basis"):
            try:
                out = layer(x_in, return_activations=True)
                if isinstance(out, (tuple, list)) and len(out) >= 3:
                    phi, basis, coef = out[0], out[1], out[2]
                    if torch.is_tensor(basis) and basis.dim() == 4:
                        return "basis", phi, basis, coef
                # fallthrough
            except TypeError:
                if mode == "basis":
                    raise  # user forced basis but it's not supported
        # fallback
        phi = layer(x_in)
        return "output", phi, None, None

    # Decide how many basis plots (shared across both layers for clean layout)
    nb_list = [getattr(l, "num_basis", None) for _, l in layers]
    have_basis = all(nb is not None for nb in nb_list)
    if mode == "basis" and not have_basis:
        raise RuntimeError("mode='basis' requested but some layers do not expose num_basis / basis tensor.")

    # If plotting per-basis, use min across layers; else 1 column per layer
    if have_basis and mode != "output":
        num_basis_to_show = min(max_basis, min(int(nb) for nb in nb_list))
        n_cols = num_basis_to_show
    else:
        num_basis_to_show = 1
        n_cols = 1

    n_rows = len(layers)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    for r, (name, layer) in enumerate(layers):
        _reset(layer)

        layer_in = getattr(layer, "in_dim", 1)
        layer_out = getattr(layer, "out_dim", 1)
        i_idx = int(np.clip(in_dim_idx, 0, layer_in - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer_out - 1))

        for c in range(n_cols):
            b_idx = c
            ax = axes[r, c]

            P_up, P_dn = [], []
            weight_str = ""
            kind_used = None

            # up sweep
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    kind, phi, basis, coef = _try_call(layer, x_in)
                kind_used = kind
                if kind == "basis":
                    P_up.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    # effective output response
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_up.append(phi[0, o_idx].item())
                    else:
                        P_up.append(phi.reshape(-1)[0].item())

            # down sweep
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    kind, phi, basis, coef = _try_call(layer, x_in)
                if kind == "basis":
                    P_dn.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_dn.append(phi[0, o_idx].item())
                    else:
                        P_dn.append(phi.reshape(-1)[0].item())

            # coef annotation if available
            if kind_used == "basis":
                try:
                    w = float(coef[0, i_idx, b_idx].item())
                    weight_str = f", w={w:.3f}"
                except Exception:
                    weight_str = ""

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2, label="Up")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Down")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization / Response")

            if kind_used == "basis":
                w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)
                ax.set_title(f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}", fontsize=10)
            else:
                ax.set_title(f"{name} (effective output)", fontsize=11)

            if c == 0:
                ax.legend(fontsize=9)

        # Hide extra columns if we fell back to output mode for this layer
        if n_cols > 1 and kind_used != "basis":
            for c in range(1, n_cols):
                axes[r, c].set_visible(False)


    fig.suptitle(
        "KanFEPA-MLP-NODE Ferroelectric Hysteresis Loops",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f"KanFEPA_MLP_NODE_Hysteresis/kanFEPA-MLP-NODE_hysteresis_epoch{epoch}.png")
    #plt.show()


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def ferro_params(layer, i_idx, o_idx, b_idx):
    # Works for BOTH FerroelectricBasis and BatchedFerroelectricBasis you showed
    w  = float(layer.coef[i_idx, o_idx, b_idx].detach().cpu())
    Ps = float(layer.Ps [i_idx, o_idx, b_idx].detach().cpu())
    Ec = float(layer.Ec [i_idx, o_idx, b_idx].detach().cpu())
    k  = float(layer.k  [i_idx, o_idx, b_idx].detach().cpu())
    return w, Ps, Ec, k


def visualize_all_ferro_bases_KanFEPA_MLP_NODE_noisy(
    model,
    epoch,
    field_range=(-5, 5),
    n_points=200,
    max_basis=6,
    which_in_out=(0, 0),
    mode="auto",          # "auto" | "basis" | "output"
    share_x=True,
    noise_std=0.2,       # override layer.noise_std if provided
    seed=0,               # deterministic noise for visualization
    save_dir="Noisy_KanFEPA_MLP_NODE_Hysteresis",
):
    model.eval()
    device = next(model.parameters()).device

    layers = [
        ("odefunc.fc1", model.odefunc.fc1),
        ("odefunc.fc2", model.odefunc.fc2),
    ]

    # deterministic noise
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    E_up = torch.linspace(field_range[0], field_range[1], n_points, device=device)
    E_dn = torch.linspace(field_range[1], field_range[0], n_points, device=device)

    in_dim_idx, out_dim_idx = which_in_out

    def _reset(layer):
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    def _make_input(layer, e_scalar):
        in_dim = getattr(layer, "in_dim", 1)
        return torch.full((1, in_dim), float(e_scalar), dtype=torch.float32, device=device)

    def _try_call(layer, x_in):
        """
        Returns (kind, phi, basis)
          kind: "basis" if we got per-basis tensor; otherwise "output"
        """
        if mode == "output":
            phi = layer(x_in)
            return "output", phi, None

        if mode in ("auto", "basis"):
            try:
                out = layer(x_in, return_activations=True)
                if isinstance(out, (tuple, list)) and len(out) >= 3:
                    phi, basis = out[0], out[1]
                    if torch.is_tensor(basis) and basis.dim() == 4:
                        return "basis", phi, basis
            except TypeError:
                if mode == "basis":
                    raise

        phi = layer(x_in)
        return "output", phi, None

    # layout choice
    nb_list = [getattr(l, "num_basis", None) for _, l in layers]
    have_basis = all(nb is not None for nb in nb_list)

    if mode == "basis" and not have_basis:
        raise RuntimeError("mode='basis' requested but some layers do not expose num_basis / basis tensor.")

    if have_basis and mode != "output":
        num_basis_to_show = min(max_basis, min(int(nb) for nb in nb_list))
        n_cols = num_basis_to_show
    else:
        num_basis_to_show = 1
        n_cols = 1

    n_rows = len(layers)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.8 * n_rows),
        sharex=share_x
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    for r, (name, layer) in enumerate(layers):
        # ---- force noisy mode for this layer (and restore after) ----
        old_use = getattr(layer, "use_noise", False)
        old_std = getattr(layer, "noise_std", None)

        if hasattr(layer, "use_noise"):
            layer.use_noise = True
        if noise_std is not None and hasattr(layer, "noise_std"):
            layer.noise_std = float(noise_std)

        _reset(layer)

        layer_in = getattr(layer, "in_dim", 1)
        layer_out = getattr(layer, "out_dim", 1)
        i_idx = int(np.clip(in_dim_idx, 0, layer_in - 1))
        o_idx = int(np.clip(out_dim_idx, 0, layer_out - 1))

        kind_used = None

        for c in range(n_cols):
            b_idx = c
            ax = axes[r, c]

            P_up, P_dn = [], []

            # up sweep
            for e in E_up:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    kind, phi, basis = _try_call(layer, x_in)
                kind_used = kind
                if kind == "basis":
                    P_up.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_up.append(phi[0, o_idx].item())
                    else:
                        P_up.append(phi.reshape(-1)[0].item())

            # down sweep
            for e in E_dn:
                x_in = _make_input(layer, e.item())
                with torch.no_grad():
                    kind, phi, basis = _try_call(layer, x_in)
                if kind == "basis":
                    P_dn.append(basis[0, i_idx, o_idx, b_idx].item())
                else:
                    if phi.dim() == 2 and phi.size(1) > o_idx:
                        P_dn.append(phi[0, o_idx].item())
                    else:
                        P_dn.append(phi.reshape(-1)[0].item())

            ax.plot(E_up.detach().cpu().numpy(), np.array(P_up), 'b-', linewidth=2,
                    label=f"Noisy up (std={getattr(layer, 'noise_std', 0.0):.3f})")
            ax.plot(E_dn.detach().cpu().numpy(), np.array(P_dn), 'r-', linewidth=2, label="Noisy down")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Electric Field (E)")
            ax.set_ylabel("Polarization / Response")

            if kind_used == "basis":
                w, Ps, Ec, k = ferro_params(layer, i_idx, o_idx, b_idx)
                ax.set_title(
                    f"{name}\nBasis b={b_idx} | w={w:.3f}, Ps={Ps:.3f}, Ec={Ec:.3f}, k={k:.3f}",
                    fontsize=10
                )
            else:
                ax.set_title(f"{name} (effective output)", fontsize=11)

            if c == 0:
                ax.legend(fontsize=9)

        # Hide extra columns if we fell back to output mode for this layer
        if n_cols > 1 and kind_used != "basis":
            for c in range(1, n_cols):
                axes[r, c].set_visible(False)

        # restore layer settings
        if hasattr(layer, "use_noise"):
            layer.use_noise = old_use
        if hasattr(layer, "noise_std"):
            layer.noise_std = old_std

    fig.suptitle(
        f"20% Noise Added: KanFEPA-MLP-NODE Ferroelectric Hysteresis Loops",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"kanFEPA-MLP-NODE_hysteresis_noisy_epoch{epoch}.png")
    plt.savefig(out_path, dpi=200)
    #plt.show()

    return out_path




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
            '''
            visualize_all_ferro_bases_KanFEPA_MLP_NODE(
                model,
                ep,
                field_range=(-5, 5),
                n_points=200,
                max_basis=6,
                which_in_out=(0, 0),
                mode="auto",   # try per-basis if available; otherwise effective output
            )
            '''

            _ = visualize_all_ferro_bases_KanFEPA_MLP_NODE_noisy(
                model, epoch=ep, noise_std=0.2, seed=0, max_basis=6
            )
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

    EPOCHS = 100 #this is the sweet spot, if epochs>100, the acc of FEPA-RNN goes down afterwards

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


    print("Training KanFEPA-RNN...")
    base_rnn_model, base_rnn_train_acc, base_rnn_test_acc = train_KAN_RNN(epochs=EPOCHS)





    print("Training KanFEPA-RNN-NODE (Latent Space Embedded)...")

    
    rnn_node_model, rnn_node_train_acc, rnn_node_test_acc = train_rnn_node(
    train_path="data/ECG200_TRAIN.txt",
    test_path="data/ECG200_TEST.txt",
    batch_size=1,
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
   


    print("Training Baseline Digital RNN...")
    digital_rnn_model, digital_rnn_train_loss, digital_rnn_test_acc = train_rnn_baseline(
        train_path="data/ECG200_TRAIN.txt",
        test_path="data/ECG200_TEST.txt",
        batch_size=8,
        epochs=EPOCHS,
        lr=1e-5,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1, #1 layer is better than deeper layers
        dropout=0.0,
        bidirectional=True,
        clip_grad=1.0,
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
    plt.title("Training Loss (No Noise or Pertubations)", fontsize=30)

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
    plt.title("Test Accuracy (No Noise or Pertubations)", fontsize=30)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
