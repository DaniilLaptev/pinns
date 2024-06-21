
import torch

from pinns.models import PINN
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, model, scheduler = None, **hyperparameters):
        
        if isinstance(model, torch.nn.Module):
            parameters = model.parameters()
        elif isinstance(model, PINN):
            parameters = model.model.parameters()
            
        self.optim = torch.optim.Adam(parameters, **hyperparameters)
        self.mode = 'backward'
        
        self.scheduler = self.get_scheduler(scheduler)
        
    def step(self):
        self.optim.step()
        
    def clear_cache(self):
        self.optim.zero_grad()