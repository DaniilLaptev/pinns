
import torch.nn as nn

from .model import PINN

class FF(PINN):
    def __init__(self, layers, activ=nn.ReLU(), biases=True, dropout = 0):
        
        modules = [nn.Linear(layers[0], layers[1])]
            
        for i in range(1, len(layers) - 1):
            modules.append(activ)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layers[i], layers[i + 1], bias=biases))
            
        self.model = nn.Sequential(*modules)
        
    def train(self):
        for module in self.model.modules():
            module.train()
    
    def eval(self):
        for module in self.model.modules():
            module.eval()