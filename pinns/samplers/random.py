
import torch

from .sampler import Sampler

from collections import defaultdict

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
    
class RandomDictSampler(Sampler):
    def __init__(self, domain, num_pts):
        self.domain = domain
        self.num_pts = num_pts
    
    def __call__(self):
        
        pts = defaultdict()
        for k, v in self.domain.items():
            pts[k] = torch.rand((self.num_pts, 1)) * (v[1] - v[0]) + v[0]
            pts[k].requires_grad = True
        
        return pts