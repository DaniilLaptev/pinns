
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import grad

from modules.utils import l2

class Diffusion:
    def __init__(self, D, boundaries, resolution, b_values):
        self.D = D
        self.L, self.R, self.T = boundaries
        self.left_boundary, self.right_boundary, self.initial_values = b_values
        
        self.Nt, self.Nx = resolution
        self.solution = self._solve()
        self.domain = torch.cartesian_prod(
            torch.linspace(self.L, self.R, self.Nx),
            torch.linspace(0, self.T, self.Nt)
            )
        
        self.left_values = self.right_values = self.init_values = None
        
    def error(self, predictions):
        return float(l2(predictions.reshape(self.Nx, self.Nt).T, self.solution))
    
    def test(self, model):
        predictions = model(self.domain).detach().cpu()
        return predictions
    
    def boundary_loss(self, model, pts, lbcoefs):
        
        x_init, t_left, t_right = pts
        init =  model(torch.vstack([x_init, torch.zeros_like(x_init)]).T)
        left =  model(torch.vstack([torch.ones_like(t_left)  * self.L, t_left]).T)
        right = model(torch.vstack([torch.ones_like(t_right) * self.R, t_right]).T)
        
        init_error = torch.square(self.init_values - init.flatten()).mean()
        left_error = torch.square(self.left_values - left.flatten()).mean()
        right_error = torch.square(self.right_values - right.flatten()).mean()
        
        return lbcoefs[0] * init_error + lbcoefs[1] * left_error + lbcoefs[2] * right_error
    
    def inner_loss(self, model, pts, lfcoefs):
    
        x, t = pts
        u = model(torch.hstack([x, t]))
        
        ut =  grad(u,  t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
        ux =  grad(u,  x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        uxx = grad(ux, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        loss = torch.mean(torch.square(ut - self.D * uxx))
        return lfcoefs[0] * loss
        
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

            b = B.dot(u[n, :])
            b[0] = alpha * self.left_boundary[n]
            b[-1] = alpha * self.right_boundary[n]

            u[n+1, :] = A_reversed @ b

        return u
    
    def plot(self, predictions, fig, show=False):
        
        X = self.domain[:, 0].reshape(self.Nx, self.Nt).T
        T = self.domain[:, 1].reshape(self.Nx, self.Nt).T
        ax = fig.add_subplot(111, projection='3d')
        
        if predictions is None:
            ax.plot_surface(X, T, self.solution, cmap='viridis')
        else:
            ax.plot_surface(X, T, predictions.reshape(self.Nx, self.Nt).T.numpy(), cmap='viridis')
        
        if show:
            plt.show()
        return fig
    
    @staticmethod
    def get_problem(N):
        match N:
            case 1:
                D = 0.5
                L, R, T = 0, 1, 0.5
                Nt, Nx = 500, 150
                left_boundary = right_boundary = torch.zeros(Nt)
                initial_conditions = torch.sin(2 * np.pi * torch.linspace(L, R, Nx)) ** 2
                
            case 2:
                D = 0.1
                L, R, T = -2, 2, 40
                Nt, Nx = 500, 150
                left_boundary = right_boundary = torch.zeros(Nt)
                initial_conditions = torch.sin(0.5 * np.pi * torch.linspace(L, R, Nx)) ** 2
                
            case 3:
                def f(x):
                    return torch.exp(torch.sin(x)*torch.log(1/(x + np.finfo(float).eps)))
                D = 0.4
                L, R, T = 0, 14, 15
                Nt, Nx = 500, 150
                left_boundary = right_boundary = torch.zeros(Nt)
                initial_conditions = f(torch.linspace(L, R, Nx))
                
            case 4:
                D = 0.7
                L, R, T = -1, 5, 5
                Nt, Nx = 500, 250
                left_boundary = right_boundary = torch.zeros(Nt)
                initial_conditions = torch.sin(torch.exp(-torch.linspace(L, R, Nx) + 0.145))*5
                
            case 5:
                D = 0.5
                L, R, T = 0, 1, 0.5
                Nt, Nx = 500, 750
                left_boundary = torch.sin(4 * torch.pi * torch.linspace(0, T, Nt))
                right_boundary = torch.zeros(Nt)
                initial_conditions = torch.sin(np.pi * torch.linspace(L, R, Nx)) ** 2
                
        problem = Diffusion(
            D, (L, R, T), (Nt, Nx),
            (left_boundary, right_boundary, initial_conditions)
        )
        return problem
    
