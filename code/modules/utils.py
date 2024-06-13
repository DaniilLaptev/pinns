import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn
from scipy.fft import fftfreq

import imageio
from PIL import Image

import os
import glob

def clear_temp():
    files = glob.glob('./.temp/*')
    for f in files:
        os.remove(f)

def l2(predicts, target):
    return np.sqrt(np.square(predicts - target).sum())

def rmse(predicts, target):
    return np.sqrt(np.square(predicts - target).mean())

def create_animation(images, path, duration=5, fps=60, loop=0, type='mp4'):
    
    if type == 'mp4':
        writer = imageio.get_writer(path, fps=fps)

        for im in images:
            writer.append_data(imageio.imread(im))
        writer.close()
        
    if type == 'gif':
        imgs = [Image.open(filename) for filename in images]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, fps=fps, loop=loop)
    
def plot_frequencies(domain, sol, fsol, names=None, size=(5, 5)):
    N = domain.shape[0]
    if names is None:
        names = [None for f in sol]
    
    fig, axs = plt.subplots(2, 1, figsize=size)
    
    for f, name in zip(sol, names):
        axs[0].plot(domain, f, label=name)
    axs[0].legend()
    
    dt = np.diff(domain)[0]
    for ff, name in zip(fsol, names):
        freqs = fftfreq(len(ff), dt)
        axs[1].plot(freqs[:N//2], np.abs(ff)[:N//2], label=name)
    axs[1].set_xscale('log')
    axs[1].legend()
    
    plt.show()

class FeedForwardNetwork(nn.Module):
    def __init__(
        self, 
        hidden_layers, hidden_dim, 
        input_dim=1, output_dim=1, 
        activation=nn.Tanh(),
        ):
        super(FeedForwardNetwork, self).__init__()
        
        self.L = hidden_layers
        self.W = hidden_dim
        
        self.model = nn.Sequential()
        self.activation = activation
        
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
                    
    def get_norms(model):
        norms = {}
        for name, param in model.model.named_parameters():
            weight = param.detach().norm(p=2).item()
            gradient = param.grad.detach().norm(p=2).item() if param.grad is not None else None
            norms[name] = [float(weight), float(gradient)]
        return norms
    
    def get_params(model):
        params = []
        for param in model.parameters():
            params.extend(param.flatten().detach())
        return torch.tensor(params)