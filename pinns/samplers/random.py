
import torch

from .sampler import Sampler

class RandomSampler(Sampler):
    def __init__(self, domain, num_pts, mode='continuous'):
        self.domain = domain
        self.num_pts = num_pts
        self.mode = mode
    
    def __call__(self):
        
        if self.mode == 'continuous':
            a, b = self.domain
            pts = torch.rand(*(self.num_pts), requires_grad=True) * (b - a) + a
        
        else:
            # todo: if mode is 'array', we will randomly choose domain's subset.
            raise NotImplementedError(f'Mode {self.mode} is not implemented yet.')
        
        return pts