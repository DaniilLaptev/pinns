
from .autograd import _autograd_derivative
from .findiff import _finite_diff_derivative

class Derivative:
    def __init__(self, method = 'autograd', model = None):
        self.method = method
        
        METHODS = {
            'autograd': _autograd_derivative,
            # 'analytical': _analytical_derivative,
            'findiff': _finite_diff_derivative
        }
        
        if callable(method):
            self._d = method
        elif method not in METHODS.keys():
            raise NotImplementedError("Provided method is not implemented.")
        else:
            self._d = METHODS[method]
            
        if method in ['findiff', 'analytical'] and model is None:
            if model is None:
                raise ValueError('You must provide model to calculate finite differences.')
        
        self.f = model
        
    def __call__(self, y, x, orders = 1, **method_args):
        
        if isinstance(orders, int):
            orders = [orders]
        
        if self.method == 'autograd':
            derivs = _autograd_derivative(y, x, orders)
            
        if self.method == 'findiff':
            derivs = _finite_diff_derivative(self.f, y, x, orders, **method_args)
            
        # if self.method == 'analytical':
        #     derivs = _analytical_derivative(y, x, orders, **method_args)
            raise 
            
        return derivs[0] if len(derivs) == 1 else derivs