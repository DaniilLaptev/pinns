
import torch
from .sampler import Sampler
from ..fancytensor import FancyTensor

class ConstantSampler(Sampler):
    def __init__(self, pts):
            
        if isinstance(pts, FancyTensor):
            self.pts = pts
        
        elif isinstance(pts, torch.Tensor):
            self.pts = FancyTensor(pts, requires_grad=pts.requires_grad)
            
        elif isinstance(pts, (list, tuple)):
            self.pts = [
                FancyTensor(p, p.requires_grad) for p in pts 
                if isinstance(p, torch.Tensor)
                ]
            assert len(self.pts) == len(pts)
            
        else:
            raise TypeError('Please provide tensor or list of tensors.')
    
    def __call__(self):
        return self.pts