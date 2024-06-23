
import torch

from .sampler import Sampler
from ..fancytensor import FancyTensor

class RandomRectangularSampler(Sampler):
    def __init__(self, domain, num_pts, requires_grad = True, names = None):
        
        
        self.num_pts = num_pts
        
        self.names = names
        if isinstance(domain, dict) and names is None:
            self.names = list(domain.keys())
            self.domain = torch.tensor(list(domain.values())).T
            
        elif isinstance(domain, (list, tuple)):
            self.domain = torch.tensor(domain).T
    
    def __call__(self):
        
        a, b = self.domain[0], self.domain[1]
        pts = torch.rand(self.num_pts, len(self.domain)) * (b - a) + a
        
        return FancyTensor(pts, names = self.names)