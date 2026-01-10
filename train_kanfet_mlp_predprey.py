import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import os
import gc
import torch.nn as nn
import sys
sys.path.append("efficient_kan/")
import efficientkan #from efficient kan
from kan_diffusion.kan import KAN


#Generate LV predator-prey data
#dx/dt=alpha*x-beta*x*y
#dy/dt=delta*x*y-gamma*y

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

    plt.legend(['x_data', 'y_data', 'x_KanFet_MLP', 'y_KanFet_MLP'])
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


def kan_plotter(plot_name, pred, soln_arr, epoch, loss_train, loss_test):
    #callback plotter during training, plots current solution
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KanFet_MLP', 'y_KanFet_MLP'])
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

    plt.legend(['x_data', 'y_data', 'x_KanFet_MLP', 'y_KanFet_MLP'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/optimal/train_trial_.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')

    plt.close('all')


def kan_plotter_opt(plot_name, pred, soln_arr, epoch, loss_train, loss_test):
    #plots the optimal solution 
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KanFet_MLP', 'y_KanFet_MLP'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/optimal/train_trial_.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')

    plt.close('all')

            

#instead of integrating precisely, we use the model to learn the next state 
def rollout(model, X0, steps):
    X = X0
    traj = [X]
    for _ in range(steps):
        X = model(X)
        traj.append(X)
    return torch.stack(traj, dim=0)


class ResidualBottleneckMLPHead(nn.Module):
    def __init__(self, d_out: int, bottleneck: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_out, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_out),
            nn.Dropout(dropout),
        )
    def forward(self, y):
        return y + self.net(y)


class KANFET_ODE_WithHead(nn.Module):
    """
    ODE dynamics: dz/dt = f(z)  where f is KANFET (unchanged).
    Head is applied AFTER odeint to the predicted trajectory, not inside f.
    """
    def __init__(self, kanfet: nn.Module, state_dim: int, head_bottleneck: int = 32, head_dropout: float = 0.0):
        super().__init__()
        self.kanfet = kanfet
        self.head = ResidualBottleneckMLPHead(state_dim, bottleneck=head_bottleneck, dropout=head_dropout)

    def rhs(self, X):
        # IMPORTANT: no head here (keeps ODE fast)
        dX = self.kanfet(X)    # base dynamics
        dX = self.head(dX)     # refine ΔX
        return dX


loss_list_train=[]
loss_list_test=[]



loss_min=1e10 #arbitrarily large to overwrite later
opt_plot_counter=0

epoch_cutoff=10 #start at smaller lr to initialize, then bump it up


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kan_fet_core = efficientkan.KANFET(layers_hidden=[2,10,2], grid_size=5).to(device)
model = KANFET_ODE_WithHead(kan_fet_core, state_dim=2, head_bottleneck=32, head_dropout=0.0).to(device)

# ---- tensors to device + consistent dtype ----
X0=torch.unsqueeze((torch.Tensor(np.transpose(X0))), 0)
X0.requires_grad=True

soln_arr = torch.tensor(soln_arr, dtype=torch.float32).to(device)
soln_arr.requires_grad_(True)   # (usually you DON'T need grad on targets; ok to remove)
soln_arr_train = soln_arr[:N_t_train, :]  # (T_train, 2)

t_learn = torch.tensor(np.linspace(0, tf_learn, N_t_train), dtype=torch.float32).to(device)
t = torch.tensor(t, dtype=torch.float32).to(device)

# optimizer should see both KAN + head
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if is_restart:
    # If you only saved core KANFET ckpt previously, you can load it into core:
    # kan_fet_core.load_ckpt('ckpt_predprey')
    # If you have a full wrapper ckpt, do torch.load.
    kan_fet_core.load_ckpt('ckpt_predprey')

kan_model_loss_list_train = []
kan_model_loss_list_test = []

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    # ---- ODE rollout using RHS (KANFET only) ----
    pred = rollout(model.rhs, X0, len(t_learn) - 1)
    #pred = torchodeint(model.rhs, X0, t_learn)  # typical shape: (T_train, 1, 2)

    # ---- apply head AFTER odeint (Option B) ----
    pred = model.head(pred)  # same shape as pred

    # Align shapes: pred[:,0,:] => (T_train, 2)
    loss_train = torch.mean((pred[:, 0, :] - soln_arr_train) ** 2)

    loss_train.backward()
    optimizer.step()

    kan_model_loss_list_train.append(loss_train.detach().cpu())

    # ---- test ----
    with torch.no_grad():
        pred_test = rollout(model.rhs, X0, len(t) - 1)
        #pred_test = torchodeint(model.rhs, X0, t)     # (T, 1, 2)
        pred_test = model.head(pred_test)             # Option B head
        test_loss = torch.mean((pred_test[N_t_train:, 0, :] - soln_arr[N_t_train:, :]) ** 2)

    kan_model_loss_list_test.append(test_loss.detach().cpu())
    if loss_train<loss_min:
        loss_min=loss_train
        #model.save_ckpt('ckpt_predprey_opt')
        if opt_plot_counter>=200:
            print('plotting optimal model')
            kan_plotter_opt("kan_fet_mlp_plots", pred_test[:, 0, :], soln_arr, epoch, kan_model_loss_list_train, kan_model_loss_list_test)
            opt_plot_counter=0

    print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
    ##########
    #########################make a checker that deepcopys the best loss into, like, model_optimal
    #########
    ######################and then save that one into the file, not just whatever the current one is
    if epoch % plot_freq ==0:
        #model.save_ckpt('ckpt_predprey')
        kan_plotter("kan_fet_mlp_plots", pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)

breakpoint()
