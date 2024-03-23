import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def rmse(predicts, target):
    return np.sqrt(np.square(predicts - target).mean())

def plot_ode(domain, solutions, predicted=None, size=(5, 5), title=None, save=False, show=True, dpi=300):
       
    fig = plt.figure(figsize=size)

    for solution, name in solutions:
        plt.plot(domain, solution, label=name)
        
    if predicted is not None:
        for prediction, name in predicted:
            plt.plot(domain, prediction, label=name)
    
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