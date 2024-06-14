
import torch
import numpy as np
import matplotlib.pyplot as plt

class Diffusion1d:
    def __init__(self, A, B, T, D, Nt, Nx, f, g1, g2,):
        
        # Physical properties
        self.D = D
        
        # Domain properties
        self.A, self.B, self.T = A, B, T
        self.Nt, self.Nx = Nt, Nx
        self.t = torch.linspace(0, self.T, self.Nt)
        self.x = torch.linspace(self.A, self.B, self.Nx)
        
        # Problem setting
        self.f, self.g1, self.g2 = f, g1, g2
        self.initial_values = self.f(self.x)
        self.left_boundary = self.g1(self.t)
        self.right_boundary = self.g2(self.t)

    def solve_1d(self, plot = True):

        solution = self._crank_nicholson()
        
        if plot:
            x, t = torch.meshgrid(self.x, self.t)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, t, solution.T, cmap='viridis')
            plt.show()
            
        return solution
        
    def _crank_nicholson(self):
        
        Nx, Nt = self.Nx, self.Nt
        
        dx = (self.B - self.A) / Nx
        dt = self.T / Nt
        alpha = self.D * dt / (2 * dx ** 2)

        u = torch.zeros((Nt, Nx))

        u[0, :] = self.initial_values
        u[:, 0], u[:, -1] = self.left_boundary, self.right_boundary

        A_cs = torch.diag((1 + 2*alpha) * torch.ones(Nx)) + torch.diag(-alpha * torch.ones(Nx-1), 1) + torch.diag(-alpha * torch.ones(Nx-1), -1)
        B_cs = torch.diag((1 - 2*alpha) * torch.ones(Nx)) + torch.diag(alpha * torch.ones(Nx-1), 1) + torch.diag(alpha * torch.ones(Nx-1), -1)
            
        A_reversed = torch.linalg.inv(A_cs)

        for n in range(0, Nt - 1):

            b = B_cs @ (u[n, :])
            b[0] = alpha * self.left_boundary[n]
            b[-1] = alpha * self.right_boundary[n]

            u[n+1, :] = A_reversed @ b
        
        return u

    def save(self, u = None, path = './data/'):
        init = torch.vstack([torch.zeros_like(self.x), self.x, self.initial_values]).T
        left = torch.vstack([self.t, torch.ones_like(self.t) * self.A, self.left_boundary]).T
        right= torch.vstack([self.t, torch.ones_like(self.t) * self.B, self.right_boundary]).T
        
        t_pts, x_pts = torch.meshgrid(self.t, self.x)
        if u is None:
            u = self._crank_nicholson()
        solution = torch.stack([t_pts.flatten(), x_pts.flatten(), u.flatten()]).T

        np.save(path + 'init.npy', init)
        np.save(path + 'left.npy', left)
        np.save(path + 'right.npy', right)
        np.save(path + 'solution.npy', solution)
        
    def get_random_subset(self, N_I, N_A, N_B):
        init_pts = torch.rand((N_I)).sort().values * (self.B - self.A) + self.A
        left_pts = torch.rand((N_A)).sort().values * self.T
        right_pts = torch.rand((N_B)).sort().values * self.T

        init_data = self.f(init_pts)
        left_data = self.g1(left_pts)
        right_data = self.g2(right_pts)
        
        points = [
            torch.vstack([torch.zeros_like(init_pts), init_pts]).T,
            torch.vstack([left_pts, torch.ones_like(left_pts) * self.A]).T,
            torch.vstack([right_pts, torch.ones_like(right_pts) * self.B]).T,
        ]

        targets = [
            init_data.reshape(-1, 1), 
            left_data.reshape(-1, 1), 
            right_data.reshape(-1, 1)
        ]
        
        return (points, targets)