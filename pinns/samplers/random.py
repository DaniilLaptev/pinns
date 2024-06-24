
import torch

from .sampler import Sampler

class RandomRectangularSampler(Sampler):
    def __init__(self, domain, num_pts, names = None, return_dict = True):
        self.num_pts = num_pts
        
        self.names = names
        if isinstance(domain, dict) and names is None:
            self.domain = torch.tensor(list(domain.values())).T
            self.names = list(domain.keys())
            
        elif isinstance(domain, (list, tuple)):
            self.domain = torch.tensor(domain).T
        
        self.return_dict = return_dict
    
    def __call__(self):
        
        a, b = self.domain[0], self.domain[1]
        num_vars = self.domain.shape[1]
        
        if self.return_dict:
            if self.names is None:
                raise AttributeError('Names not found for dictionary.')
            pts = {}
            for k, d in zip(self.names, self.domain.T):
                pts[k] = torch.rand((self.num_pts, 1)) * (d[1] - d[0]) + d[0]
                pts[k].requires_grad = True
        
        else:
            pts = torch.rand(self.num_pts, num_vars, requires_grad=True) * (b - a) + a
        
        return pts
        