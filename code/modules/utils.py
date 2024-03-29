import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn

def rmse(predicts, target):
    return np.sqrt(np.square(predicts - target).mean())

def plot_ode(domain, solutions, predicted=None, size=(5, 5), title=None, save=False, show=True, dpi=300):
       
    fig = plt.figure(figsize=size)

    for solution, name, color in solutions:
        plt.plot(domain, solution, label=name, color=color)
        
    if predicted is not None:
        for prediction, name, color in predicted:
            plt.plot(domain, prediction, label=name, color=color)
    
    plt.xlabel('t')
    plt.ylabel('Value')

    plt.legend()
    plt.title(title)
    
    if save:
        plt.savefig(f'./images/{title}.img', dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_pde(x, t, u, size=(5, 5), title=None, save=False, show=True, dpi=300):
    
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=size)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, T, u, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(u, origin='lower', aspect='auto', cmap='viridis', 
              extent=[x.min(), x.max(), t.min(), t.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    plt.tight_layout()
    
    if save:
        plt.savefig(f'./images/{title}.png', dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_losses(t, losses, errors, title=None, save=False, dpi=300):
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), gridspec_kw={'width_ratios': [2, 1]})

    for loss, name in losses:
        ax[0].plot(t, loss, label=name)
    for error, name in errors:
        ax[1].plot(t, error, label=name)

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    # plt.grid()
    ax[0].legend()
    ax[1].legend()
    
    ax[0].title.set_text('Loss Dynamics')
    ax[1].title.set_text('RMSE Dynamics')

    plt.tight_layout()

    if save:
        plt.savefig(f'./images/{title}.png', dpi=dpi)

    plt.show()

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_layers, hidden_dim, input_dim=1, output_dim=1):
        super(FeedForwardNetwork, self).__init__()
        
        self.L = hidden_layers
        self.W = hidden_dim
        
        self.model = nn.Sequential()
        self.activation = nn.Tanh()
        
        inp_linear = nn.Linear(input_dim, hidden_dim)
        out_linear = nn.Linear(hidden_dim, output_dim)
        
        self.model.add_module('input', inp_linear)
        self.model.add_module('activ0', self.activation)
        for i in range(hidden_layers - 1):
            linear = nn.Linear(hidden_dim, hidden_dim)
            self.model.add_module(f'linear{i+1}', linear)
            self.model.add_module(f'activ{i+1}', self.activation)
        self.model.add_module('output', out_linear)
        
    def forward(self, x):
        return self.model(x)
    
    def init_weights(self, init_rule, args=None):
        for param in self.model.parameters():
            if len(param.shape) > 1:
                if args is not None:
                    init_rule(param.data, *args)
                else:
                    init_rule(param.data)
                    
def get_corr(dataframe):

    grouped = dataframe.groupby(['init'])
    num_rows = len(grouped)

    correlation = np.zeros((num_rows, len(dataframe.columns) - 2))
    rmse_data = np.zeros((num_rows, 2))

    print(f'Instances of group...')
    for i, group in enumerate(grouped):
        group_name, group_data = group
        print(f'{group_name[0]} - {len(group_data)}')
        
        correlation[i] += group_data.drop(['init'], axis=1).corr()['rmse'][1:]

        rmse_data[i,0] += group_data['rmse'].mean()
        rmse_data[i,1] += group_data['rmse'].std()
    
    return correlation, rmse_data, dataframe.columns.drop(['init']).to_list()[1:], grouped.groups.keys()

def plot_rmse_corr(correlation, rmse_data, xlabels, ylabels, k=1):
    
    fig = plt.figure(figsize=(8 / k, 2 / k))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 2])

    p_corr = plt.subplot(gs[0])
    p_rmse = plt.subplot(gs[1])

    corr_matrix = p_corr.matshow(correlation, vmin=-1, vmax=1, interpolation='none', cmap='viridis')
    p_corr.title.set_text('Correlation of RMSE with other HPs')
    p_corr.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    p_corr.set_xticks(np.arange(len(xlabels)))
    p_corr.set_xticklabels(xlabels)
    p_corr.set_yticks(np.arange(len(ylabels)))
    p_corr.set_yticklabels(ylabels)
    p_corr.grid(False)
    
    plt.colorbar(corr_matrix, ax=p_corr, pad=0.02, aspect=7.5)

    for (i, j), z in np.ndenumerate(correlation):
        p_corr.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')

    rmse_matrix = p_rmse.imshow(np.stack([rmse_data[:,0], rmse_data[:,0]]).T, cmap='magma_r', vmin=0)
    p_rmse.title.set_text('RMSE')
    p_rmse.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    p_rmse.set_xticks(np.arange(2))
    p_rmse.set_xticklabels(['Mean', 'Std'])
    p_rmse.set(yticklabels=[])
    p_rmse.set(ylabel=None)
    p_rmse.grid(False)
    
    plt.colorbar(rmse_matrix, ax=p_rmse, pad=0.08, aspect=7.5)

    for (i, j), z in np.ndenumerate(rmse_data):
        p_rmse.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()