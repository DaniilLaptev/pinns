
import torch
import numpy as np

class Domain:
    def __init__(self, domain, names = None, shape = 'rectangular'):
        
        self.names = names
        if isinstance(domain, dict) and names is None:
            self.domain = torch.tensor(list(domain.values())).T
            self.names = list(domain.keys())
            
        elif isinstance(domain, (list, tuple)):
            self.domain = torch.tensor(domain).T
        
        elif isinstance(domain, np.ndarray):
            self.domain = torch.tensor(domain)
            
        self.num_vars = len(domain.keys())
        self.shape = shape
        
    def __getitem__(self, key):
        
        if isinstance(key, str):
            if self.names is None:
                raise ValueError('Names are not provided for this points.')
            if key not in self.names:
                raise AttributeError(f'There are no such variable: {key}')
            return self.domain[:, self.names.index(key)]
        
        elif isinstance(key, int) and self.domain.shape[1] > 1:
            return self.domain[:,key]
        
        else:    
            return self.domain[key]