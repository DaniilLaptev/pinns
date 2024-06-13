
import torch
from torch.autograd import grad

def _autograd_derivative(y, x):
    dy = grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy

def _analytical_derivative(y, x):
    pass

def _finite_diff_derivative(y, x):
    pass

def d(y, x, method = 'autograd'):
    
    if callable(method):
        return method(y, x)

    elif method == 'autograd':
        return _autograd_derivative(y, x)
    elif method == 'analytical':
        return _analytical_derivative(y, x)
    elif method == 'finite_diff':
        return _finite_diff_derivative(y, x)
    
    else:
        raise NotImplementedError("Provided method is not implemented.")