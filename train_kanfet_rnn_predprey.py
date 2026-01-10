import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder

tf=14
tf_learn=3.5
N_t_train=35
N_t=int((35*tf/tf_learn))
lr=2e-3
num_epochs=10000
plot_freq=100
is_restart=False


##coefficients from https://arxiv.org/pdf/2012.07244
alpha=1.5
beta=1
gamma=3
delta=1


x0=1 
y0=1 


def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    x=X[0]
    y=X[1]
    dxdt = alpha*x-beta*x*y
    dydt = delta*x*y-gamma*y
    dXdt=[dxdt, dydt]
    return dXdt

X0=np.array([x0, y0])
t=np.linspace(0, tf, N_t)

soln_arr=scipy.integrate.odeint(pred_prey_deriv, X0, t, args=(alpha, beta, delta, gamma))
def plotter(plot_name, pred, soln_arr, epoch, loss_train, loss_test):
    #callback plotter during training, plots current solution
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KanFet_RNN', 'y_KanFet_RNN'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/training_updates/train_epoch_"+str(epoch) +".png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')
    
    plt.figure()
    plt.semilogy(torch.Tensor(loss_train), label='train')
    plt.semilogy(torch.Tensor(loss_test), label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"{plot_name}/pred_prey/loss.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')


def plotter(plot_name, pred, soln_arr, epoch, loss_train, loss_test):
    #callback plotter during training, plots current solution
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KanFet_RNN', 'y_KanFet_RNN'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/training_updates/train_epoch_"+str(epoch) +".png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')
    
    plt.figure()
    plt.semilogy(torch.Tensor(loss_train), label='train')
    plt.semilogy(torch.Tensor(loss_test), label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"{plot_name}/pred_prey/loss.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')

    
def plotter_opt(plot_name, pred, soln_arr, epoch, loss_train, loss_test):
    #plots the optimal solution 
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KanFet_RNN', 'y_KanFet_RNN'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/optimal/train_trial_.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')

    plt.close('all')


# ----------- Logistic Basis Expansion -----------
class LogisticBasis(nn.Module):
    def __init__(self, in_dim, num_basis):
        super().__init__()
        self.a = nn.Parameter(torch.randn(in_dim, num_basis))
        self.b = nn.Parameter(torch.randn(in_dim, num_basis))

    def forward(self, x):  # x: (B, in_dim)
        x = x.unsqueeze(-1)  # → (B, in_dim, 1)
        return 2 / (1 + torch.exp(-self.a * (x - self.b)))  # (B, in_dim, num_basis)

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

class KANRegressor(nn.Module):
    """
    Replace classifier with regressor: hidden -> R^2 (delta x, delta y)
    """
    def __init__(self, in_dim, out_dim, num_basis):
        super().__init__()
        self.basis = LogisticBasis(in_dim, num_basis)
        self.activation = nn.Sigmoid()
        self.output = nn.Parameter(torch.randn(in_dim * num_basis, out_dim))

    def forward(self, x):
        phi = self.basis(x)
        phi = self.activation(phi)
        phi_flat = phi.view(x.shape[0], -1)
        return phi_flat @ self.output  # (B, out_dim)

class FullyNonlinearKANRNN(nn.Module):
    def __init__(self, input_size=3, seq_len=16, hidden_size=64, out_dim=2, num_basis=10, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnn_cell = FullyNonlinearKANCell(input_size, hidden_size, num_basis)
        self.dropout = nn.Dropout(dropout)
        self.head = KANRegressor(hidden_size, out_dim, num_basis)

    def forward(self, x):  # x: (B, seq_len, input_size)
        assert x.dim() == 3 and x.size(2) == self.input_size
        B = x.size(0)
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for tt in range(self.seq_len):
            x_t = x[:, tt, :]  # (B, input_size)
            h = self.rnn_cell(x_t, h)
        h = self.dropout(h)
        return self.head(h)  # (B,2)

# ----------------- Helpers: build (t,x,y) sequences -----------------
def make_txy_seq(t_scalar: torch.Tensor, xy: torch.Tensor, seq_len: int):
    """
    t_scalar: (B,) or (B,1)
    xy:       (B,2)
    return:   (B, seq_len, 3) with each step containing [t, x, y]
    """
    if t_scalar.dim() == 1:
        t_scalar = t_scalar.unsqueeze(1)  # (B,1)
    feat = torch.cat([t_scalar, xy], dim=1)          # (B,3)
    return feat.unsqueeze(1).repeat(1, seq_len, 1)   # (B,seq_len,3)

@torch.no_grad()
def rollout_rnn(model: nn.Module, x0y0: torch.Tensor, t_grid: torch.Tensor, seq_len: int):
    if isinstance(t_grid, np.ndarray):
        t_grid = torch.tensor(t_grid, device=x0y0.device, dtype=x0y0.dtype)
    model.eval()
    T = t_grid.shape[0]
    pred = torch.zeros(T, 2, device=x0y0.device, dtype=x0y0.dtype)
    pred[0] = x0y0

    for k in range(T - 1):
        tk = t_grid[k:k+1]                   # (1,)
        xy = pred[k:k+1, :]                  # (1,2)
        x_in = make_txy_seq(tk, xy, seq_len) # (1,seq_len,3)
        dxy = model(x_in)                    # (1,2) predicted delta
        pred[k + 1] = pred[k] + dxy.squeeze(0)
    return pred

# ----------------- Train -----------------
plot_name = "kan_fet_rnn_plots"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
soln_arr_train = soln_arr[:N_t_train, :] 
soln_arr = torch.tensor(soln_arr, dtype=torch.float32).to(device)
soln_arr.requires_grad_(True)   # (usually you DON'T need grad on targets; ok to remove)
soln_arr_train = soln_arr[:N_t_train, :]  # (T_train, 2)

t_learn = torch.tensor(np.linspace(0, tf_learn, N_t_train), dtype=torch.float32).to(device)
seq_len = 16  # small is fine since we repeat the same (t,x,y) across the sequence
model = FullyNonlinearKANRNN(input_size=3, seq_len=seq_len, hidden_size=64, out_dim=2, num_basis=10, dropout=0.0).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
loss_fn = nn.MSELoss()

loss_list_train = []
loss_list_test  = []

best_loss = float("inf")
best_state = None

x0y0 = torch.tensor([x0, y0], device=device, dtype=torch.float32)

# training pairs on [0, tf_learn]
train_states = soln_arr_train                        # (T_train,2)
train_t      = t_learn[:-1]                          # (T_train-1,)
train_xy     = train_states[:-1]                     # (T_train-1,2)
train_deltas = train_states[1:] - train_states[:-1]  # (T_train-1,2)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # batch all steps: input is (t_k, x_k, y_k) -> delta
    x_in = make_txy_seq(train_t, train_xy, seq_len=seq_len)  # (B=T_train-1, seq_len, 3)
    pred_delta = model(x_in)                                  # (B,2)
    loss_train = loss_fn(pred_delta, train_deltas)

    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_list_train.append(float(loss_train.detach().cpu()))

    with torch.no_grad():
        pred_full = rollout_rnn(model, x0y0, t, seq_len=seq_len)  # (T,2)
        loss_test = torch.mean((pred_full[N_t_train:] - soln_arr[N_t_train:]) ** 2)
        loss_list_test.append(float(loss_test.detach().cpu()))

    if loss_test.item() < best_loss:
        best_loss = float(loss_test.item())
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if (epoch % plot_freq) == 0:
        plotter(plot_name, pred_full, soln_arr, epoch, loss_list_train, loss_list_test)
        print(f"epoch {epoch:5d} | train {loss_list_train[-1]:.3e} | test {loss_list_test[-1]:.3e}")

# plot optimal
if best_state is not None:
    model.load_state_dict(best_state)

with torch.no_grad():
    pred_best = rollout_rnn(model, x0y0, t, seq_len=seq_len)

plotter_opt(plot_name, pred_best, soln_arr, epoch=num_epochs, loss_train=loss_list_train, loss_test=loss_list_test)