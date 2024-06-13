
import torch

import pinns

class Optimizer:
    
    def __init__(self, model = None, scheduler = None, **hyperparameters):
        self.scheduler = None
            
    def get_scheduler(self, scheduler):
        if scheduler is not None:
            sched_type, sched_args = scheduler
            if sched_args is None:
                self.scheduler = sched_type(self.optim)
            else:
                self.scheduler = sched_type(self.optim, *sched_args)
        else:
            self.scheduler = None
            
    def step(self, *args, **kwargs):
        raise NotImplementedError('Method step() is not implemented for this optimizer.')
            
    def clear_cache(self, *args, **kwargs):
        raise NotImplementedError('Method clear_cache() is not implemented for this optimizer.')