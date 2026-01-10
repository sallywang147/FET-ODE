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

    plt.legend(['x_data', 'y_data', 'x_KanFet_NODE', 'y_KanFet_NODE'])
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

    plt.legend(['x_data', 'y_data', 'x_KanFet_NODE', 'y_KanFet_NODE'])
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

    plt.legend(['x_data', 'y_data', 'x_KanFet_NODE', 'y_KanFet_NODE'])
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

    plt.legend(['x_data', 'y_data', 'x_KanFet_NODE', 'y_KanFet_NODE'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    plt.savefig(f"{plot_name}/pred_prey/optimal/train_trial_.png", dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close('all')

    plt.close('all')

            
        
# initialize KAN with grid=5
kan_fet_model = efficientkan.KANFET(layers_hidden=[2,10,2], grid_size=5) #k is order of piecewise polynomial
#kan_model = KAN(layers_hidden=[2,10,2], grid_size=5) #k is order of piecewise polynomial
#convery numpy training data to torch tensors: 
X0=torch.unsqueeze((torch.Tensor(np.transpose(X0))), 0)
X0.requires_grad=True
soln_arr=torch.Tensor(soln_arr)
soln_arr.requires_grad=True
soln_arr_train=soln_arr[:N_t_train, :]
t=torch.Tensor(t)
t_learn=torch.tensor(np.linspace(0, tf_learn, N_t_train))



def calDeriv(t, X):
    dXdt=kan_fet_model(X)
    return dXdt

def calkan_model_Deriv(t, X):
    dXdt=kan_fet_model(X)
    return dXdt

loss_list_train=[]
loss_list_test=[]

#initialize ADAM optimizer
optimizer = torch.optim.Adam(kan_fet_model.parameters(), lr=lr)
kan_model_optimizer = torch.optim.Adam(kan_fet_model.parameters(), lr=lr)
if is_restart==True:
    kan_fet_model.load_ckpt('ckpt_predprey')

loss_min=1e10 #arbitrarily large to overwrite later
opt_plot_counter=0

epoch_cutoff=10 #start at smaller lr to initialize, then bump it up

#p1=model.layers[0].spline_weight
#p2=model.layers[0].base_weight
#p3=model.layers[1].spline_weight
#p4=model.layers[1].base_weight

def rollout(model, X0, steps):
    dt = 1.0 / steps
    X = X0
    traj = [X]
    for _ in range(steps):
        dX = model(X)
        X = X + dt * dX
        traj.append(X)
    return torch.stack(traj, dim=0)
'''
#instead of integrating precisely, we use the model to learn the next state 
def rollout(model, X0, steps):
    X = X0
    traj = [X]
    for _ in range(steps):
        X = model(X)
        traj.append(X)
    return torch.stack(traj, dim=0)


for epoch in tqdm(range(num_epochs)):
    opt_plot_counter+=1
    #if epoch==epoch_cutoffs[2]:
    #    model = kan.KAN(width=[2,3,2], grid=grids[1], k=3).initialize_from_another_model(model, X0_train)
    optimizer.zero_grad()

    #pred=torchodeint(calDeriv, X0, t_learn, adjoint_params=[p1, p2, p3, p4])
    pred = rollout(kan_fet_model, X0, len(t_learn) - 1)
    loss_train=torch.mean(torch.square(pred[:, 0, :]-soln_arr_train))
    loss_train.retain_grad()
    loss_train.backward()
    optimizer.step()
    loss_list_train.append(loss_train.detach().cpu())
    #pred_test=torchodeint(calDeriv, X0, t, adjoint_params=[])
    #pred_test=torchodeint(calDeriv, X0, t)
    pred_test = rollout(kan_fet_model, X0, len(t) - 1)
    loss_list_test.append(torch.mean(torch.square(pred_test[N_t_train:,0, :]-soln_arr[N_t_train:, :])).detach().cpu())
    #if epoch ==5:
    #    model.update_grid_from_samples(X0)
    if loss_train<loss_min:
        loss_min=loss_train
        #model.save_ckpt('ckpt_predprey_opt')
        if opt_plot_counter>=200:
            print('plotting optimal model')
            plotter_opt("knnfet-plots", pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)
            opt_plot_counter=0

    print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
    ##########
    #########################make a checker that deepcopys the best loss into, like, model_optimal
    #########
    ######################and then save that one into the file, not just whatever the current one is
    if epoch % plot_freq ==0:
        #model.save_ckpt('ckpt_predprey')
        plotter("knnfet-plots", pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)
'''


kan_model_loss_list_train=[]
kan_model_loss_list_test=[]
for epoch in tqdm(range(num_epochs)):
    opt_plot_counter+=1
    #if epoch==epoch_cutoffs[2]:
    #    model = kan.KAN(width=[2,3,2], grid=grids[1], k=3).initialize_from_another_model(model, X0_train)
    kan_model_optimizer.zero_grad()

    pred=torchodeint(calDeriv, X0, t_learn)
    #pred = rollout(kan_fet_model, X0, len(t_learn) - 1)
    loss_train=torch.mean(torch.square(pred[:, 0, :]-soln_arr_train))
    loss_train.retain_grad()
    loss_train.backward()
    kan_model_optimizer.step()
    kan_model_loss_list_train.append(loss_train.detach().cpu())
    #pred_test=torchodeint(calDeriv, X0, t, adjoint_params=[])
    pred_test=torchodeint(calDeriv, X0, t)
    #pred_test = rollout(kan_fet_model, X0, len(t) - 1)
    kan_model_loss_list_test.append(torch.mean(torch.square(pred_test[N_t_train:,0, :]-soln_arr[N_t_train:, :])).detach().cpu())
    #if epoch ==5:
    #    model.update_grid_from_samples(X0)
    if loss_train<loss_min:
        loss_min=loss_train
        #model.save_ckpt('ckpt_predprey_opt')
        if opt_plot_counter>=200:
            print('plotting optimal model')
            kan_plotter_opt("kan_fet_node_plots", pred_test[:, 0, :], soln_arr, epoch, kan_model_loss_list_train, kan_model_loss_list_test)
            opt_plot_counter=0

    print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
    ##########
    #########################make a checker that deepcopys the best loss into, like, model_optimal
    #########
    ######################and then save that one into the file, not just whatever the current one is
    if epoch % plot_freq ==0:
        #model.save_ckpt('ckpt_predprey')
        kan_plotter("kan_fet_node_plots", pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)

breakpoint()
