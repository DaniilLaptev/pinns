
import torch
from scipy.special import binom

# def _finite_diff_derivative(f, y, x, orders, h=1, richardson=True, k=2):
#     '''
#     This method evaluates finite differences around given points x using
#     the central difference method for better accuracy. Optionally, it applies
#     Richardson extrapolation to improve accuracy.
    
#     Args:
#     - y (torch.Tensor): The output tensor with shape [num_samples, num_outs].
#     - x (torch.Tensor): The input tensor with shape [num_samples, num_vars].
#     - orders (list of int): The order of derivatives for each variable.
#     - h (float): Step size for finite differences.
#     - richardson (bool): Whether to use Richardson extrapolation.
#     - k (int): The number of Richardson extrapolation steps.
    
#     Returns:
#     - torch.Tensor: The nth order finite difference approximation with shape [num_outs, num_vars].
#     '''
    
#     num_samples, num_vars = x.shape
#     num_outs = y.shape[1]
    
#     def central_diff(y, x, order, var_idx, h):
#         deriv = torch.zeros_like(y)
#         for i in range(order + 1):
#             shift = torch.zeros_like(x)
#             shift[:, var_idx] = (order / 2 - i) * h
#             xs = x + shift
#             ys = f(xs)
#             deriv += (-1)**i * binom(order, i) * ys
#         return deriv / h**order
    
#     def richardson_extrapolation(y, x, order, var_idx, h, k):
#         D = torch.zeros((k, k, num_samples, num_outs), dtype=y.dtype)
        
#         for i in range(k):
#             h_i = h / (2**i)
#             D[i, 0] = central_diff(y, x, order, var_idx, h_i)
            
#             for j in range(1, i + 1):
#                 D[i, j] = (2**(order * j) * D[i, j - 1] - D[i - 1, j - 1]) / (2**(order * j) - 1)
        
#         return D[k - 1, k - 1]
    
#     derivs = torch.zeros((num_outs, num_vars))

#     for var_idx in range(num_vars):
#         for order in orders:
#             if richardson:
#                 deriv = richardson_extrapolation(y, x, order, var_idx, h, k)
#             else:
#                 deriv = central_diff(y, x, order, var_idx, h)
#             derivs[:, var_idx] += deriv.mean(dim=0)  # Mean over samples for each output
    
#     return derivs

def _finite_diff_derivative(f, y, x, orders, h=1, richardson=True, k=2):
        '''
        This method evaluates finite differences around a given point x using
        central difference method for better accuracy. Optionally, it applies
        Richardson extrapolation to improve accuracy.
        '''
    
        def central_diff(y, x, order, h):
            deriv = torch.zeros_like(y)
            for i in range(order + 1):
                xs = x + (order / 2 - i) * h
                ys = f(xs)
                deriv += (-1)**i * binom(order, i) * ys
            return deriv / h**order
        
        def richardson_extrapolation(y, x, order, h, k):
            D = torch.zeros((k, k, *y.shape), dtype=y.dtype)
            
            for i in range(k):
                h_i = h / (2**i)
                D[i, 0] += central_diff(y, x, order, h_i)
                
                for j in range(1, i + 1):
                    D[i, j] += (2**(order * j) * D[i, j - 1] - D[i - 1, j - 1]) / (2**(order * j) - 1)
            
            return D[k - 1, k - 1]
        
        derivs = []

        for order in orders:
            if richardson:
                deriv = richardson_extrapolation(y, x, order, h, k)
            else:
                deriv = central_diff(y, x, order, h)
            derivs.append(deriv)
        
        return derivs