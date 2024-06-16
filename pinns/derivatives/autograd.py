
import torch
from torch.autograd import grad

def _autograd_derivative(y, x, orders):
        
        derivs, tmp = [], y
            
        for i in range(1, orders[-1] + 1):
            
            deriv = grad(tmp, x, torch.ones_like(y), create_graph=True)[0]
            
            if i in orders:
                derivs.append(deriv)
            
            tmp = deriv
        
        return derivs