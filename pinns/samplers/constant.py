
import torch
from .sampler import Sampler

class ConstantGridSampler(Sampler):
    def __init__(self, domain, num_pts, reshape=True):
        pts = torch.linspace(domain[0], domain[1], num_pts, requires_grad=True)
        self.pts = pts.reshape(-1, 1) if reshape else pts
        
    def __call__(self):
        return self.pts
    
class ConstantSampler(Sampler):
    def __init__(self, pts):
        self.pts = pts
    
    def __call__(self):
        return self.pts