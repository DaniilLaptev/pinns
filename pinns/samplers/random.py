
import torch

from .sampler import Sampler
from ..domain import Domain

class RandomSampler(Sampler):
    def __init__(self, domain, num_pts, names = None, return_dict = True):
        self.num_pts = num_pts
        
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(domain, names)
            
        self.return_dict = return_dict
    
    def __call__(self):
        
        num_vars = self.domain.num_vars
        
        if self.return_dict:
            if self.domain.names is None:
                raise AttributeError('Names not found for dictionary.')
            pts = {}
            for k, d in zip(self.domain.names, self.domain.domain.T):
                pts[k] = torch.rand((self.num_pts, 1)) * (d[1] - d[0]) + d[0]
                pts[k].requires_grad = True
        
        else:
            a, b = self.domain.domain[0], self.domain.domain[1]
            pts = torch.rand(self.num_pts, num_vars, requires_grad=True) * (b - a) + a
        
        return pts
        