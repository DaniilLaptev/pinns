
import torch

from .sampler import Sampler

from collections import defaultdict

class RandomSampler(Sampler):
    def __init__(self, domain, num_pts, mode = 'continuous', return_dict = True):
        self.domain = domain
        self.num_pts = num_pts
        self.mode = mode
        self.return_dict = return_dict
    
    def __call__(self):
        
        pts = defaultdict()
        for k, v in self.domain.items():
            
            if self.mode == 'continuous':
                pts[k] = torch.rand((self.num_pts, 1)) * (v[1] - v[0]) + v[0]
                pts[k].requires_grad = True
        
            else:
                # todo: if mode is 'array', we will randomly choose domain's subset.
                raise NotImplementedError(f'Mode {self.mode} is not implemented yet.')
        
        if not self.return_dict:
            return torch.hstack(list(pts.values()))
        
        return pts