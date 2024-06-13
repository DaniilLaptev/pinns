
import torch

import pinns
from .optimizer import Optimizer

class LBFGS(Optimizer):
    def __init__(self, model, scheduler = None, **hyperparameters):
        
        if isinstance(model, torch.nn.Module):
            parameters = model.parameters()
        elif isinstance(model, pinns.models.model.PINN):
            parameters = model.model.parameters()
            
        self.optim = torch.optim.LBFGS(parameters, **hyperparameters)
        self.mode = 'closure'
        
        self.scheduler = self.get_scheduler(scheduler)
        
    def step(self, closure):
        return self.optim.step(closure)
        
    def clear_cache(self):
        self.optim.zero_grad()