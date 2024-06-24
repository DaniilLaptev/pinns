
import torch
from .sampler import Sampler

class ConstantSampler(Sampler):
    def __init__(self, pts):
        self.pts = pts
    
    def __call__(self):
        return self.pts