
import numpy as np

import torch
from torch.autograd import grad

from scipy.integrate import solve_ivp


class DampedHarmonicOscillator:
    def __init__(self, T, params, initial_conditions):
        
        self.T = T
        self.d, self.w0 = params
        self.init_vals = torch.tensor(initial_conditions)

        self.t = torch.linspace(0, self.T, 128)
        self.solution = self._solve()
    
    def loss_initial(self, model):
        zero = torch.tensor([0.], requires_grad=True)
        x = model(zero)
        v = grad(x, zero, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        return torch.mean(torch.square(torch.hstack([x, v]) - self.init_vals))
    
    def loss_physical(self, model, t):
        x = model(t)
        v = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        a = grad(v, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        
        return torch.mean(torch.square(a + 2*self.d*self.w0 * v + self.w0**2 * x))
    
    def _solve(self):
        def harmonic_oscillator(t, y, d, w0):
            x, v = y
            dxdt = v
            dvdt = -2*d*w0 * v - w0**2 * x
            return [dxdt, dvdt]

        solution = solve_ivp(harmonic_oscillator, 
                             (0, self.T), 
                             self.init_vals, 
                             args=(self.d, self.w0), 
                             t_eval=self.t)
        return solution.y[0]


class LotkaVolterra:
    def __init__(self, T, params, initial_conditions):
        
        self.T = T
        self.alpha, self.beta, self.delta, self.gamma = params
        self.init_vals = torch.tensor(initial_conditions)

        self.t = torch.linspace(0, self.T, 128)
        self.solution = self._solve()
    
    def loss_initial(self, model):
        zero = torch.tensor([0.], requires_grad=True)
        x = model(zero)
        return torch.mean(torch.square(x - self.init_vals))
    
    def loss_physical(self, model, t):
        xy = model(t)
        x = xy[:,[0]]
        y = xy[:,[1]]

        dX = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        dY = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        
        loss_dX = torch.mean(torch.square(dX - self.alpha * x + self.beta * x * y))
        loss_dY = torch.mean(torch.square(dY - self.delta * x * y + self.gamma * y))
        
        return loss_dX, loss_dY
    
    def _solve(self):
        def lotka_volterra(t, y, alpha, beta, delta, gamma):
            x, y = y
            dx_dt = alpha * x - beta * x * y
            dy_dt = delta * x * y - gamma * y
            return [dx_dt, dy_dt]

        solution = solve_ivp(lotka_volterra, 
                             (0, self.T),
                             self.init_vals, 
                             method='RK45',
                             args=(self.alpha, self.beta, self.delta, self.gamma), 
                             t_eval=self.t.numpy())
        
        return solution.y


class Diffusion:
    def __init__(self, D, boundaries, resolution, b_values):
        self.D = D
        self.L, self.R, self.T = boundaries
        self.left_boundary, self.right_boundary, self.initial_values = b_values
        
        self.Nt, self.Nx = resolution
        self.solution = self._solve()
        self.t = torch.linspace(0, self.T, self.Nt)
        self.x = torch.linspace(self.L, self.R, self.Nx)
        
        self.left_values, self.right_values, self.init_values = None, None, None
    
    def loss_boundary(self, model, t_left, t_right, x_init):
        left =  model(torch.vstack([torch.ones_like(t_left)  * self.L, t_left]).T)
        right = model(torch.vstack([torch.ones_like(t_right) * self.R, t_right]).T)
        init =  model(torch.vstack([x_init, torch.zeros_like(x_init)]).T)
        
        left_error = torch.square(self.left_values - left.flatten()).mean()
        right_error = torch.square(self.right_values - right.flatten()).mean()
        init_error = torch.square(self.init_values - init.flatten()).mean()
        
        return left_error + right_error, init_error
    
    def loss_physical(self, model, x, t):
        u = model(torch.hstack([x, t]))
        
        ut =  grad(u,  t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
        ux =  grad(u,  x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        uxx = grad(ux, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    
        return torch.mean(torch.square(ut - self.D * uxx))
        
    def _solve(self):
        Nt, Nx = self.Nt, self.Nx
        
        dx = (self.R - self.L) / Nx     # Spatial step size
        dt = self.T / Nt                # Time step size
        alpha = self.D * dt / (2 * dx ** 2)

        # Crank-Nicholson method
        u = np.zeros((Nt, Nx))

        # Initial condition
        u[0, :] = self.initial_values

        # Boundary conditions
        u[:, 0], u[:, -1] = self.left_boundary, self.right_boundary

        A = np.diag((1 + 2*alpha) * np.ones(Nx)) + np.diag(-alpha * np.ones(Nx-1), 1) + np.diag(-alpha * np.ones(Nx-1), -1)
        B = np.diag((1 - 2*alpha) * np.ones(Nx)) + np.diag(alpha * np.ones(Nx-1), 1) + np.diag(alpha * np.ones(Nx-1), -1)
            
        A_reversed = np.linalg.inv(A)

        for n in range(0, Nt - 1):
            b = np.dot(B, u[n, :])
            u[n+1, :] = A_reversed @ b

        return u