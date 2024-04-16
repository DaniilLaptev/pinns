
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots
plt.style.use(['science'])
mpl.rcParams["font.size"] = "12"

import torch
from torch.autograd import grad

from scipy.integrate import solve_ivp

from modules.utils import l2

class LorenzSystem:
    def __init__(self, T, params, initial_conditions):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.T = T
        self.s, self.p, self.b = params
        self.init_vals = torch.tensor(initial_conditions).to(self.device)

        self.domain = torch.linspace(0, self.T, 1024).to(self.device)
        self.solution = self._solve()
        
    def test(self, model):
        return model(self.domain.reshape(-1, 1)).detach().cpu()
        
    def error(self, predictions):
        x = predictions[:,0].flatten()
        y = predictions[:,1].flatten()
        z = predictions[:,2].flatten()
        error_x = l2(x, self.solution[0])
        error_y = l2(y, self.solution[1])
        error_z = l2(z, self.solution[2])
        return float((error_x + error_y + error_z) / 3)
    
    def boundary_loss(self, model, t, lbcoefs):
        x = model(t)
        loss = torch.mean(torch.square(x - self.init_vals))
        return lbcoefs[0] * loss
    
    def inner_loss(self, model, t, lfcoefs):
        xyz = model(t)
        x = xyz[:,[0]]
        y = xyz[:,[1]]
        z = xyz[:,[2]]

        dX = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        dY = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        dZ = grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
        
        loss_dX = torch.mean(torch.square(dX - self.s * (y - x)))
        loss_dY = torch.mean(torch.square(dY - x * (self.p - z) + y))
        loss_dZ = torch.mean(torch.square(dZ - x * y + self.b * z))
        
        return lfcoefs[0] * loss_dX + lfcoefs[1] * loss_dY + lfcoefs[2] * loss_dZ
    
    def _solve(self):
        def lorenz_system(t, u, s, p, b):
            x, y, z = u
            dx = s * (y - x)
            dy = x * (p - z) - y
            dz = x * y - b * z
            return [dx, dy, dz]

        solution = solve_ivp(lorenz_system, 
                             (0, self.T),
                             self.init_vals.cpu().numpy(), 
                             method='RK45',
                             args=(self.s, self.p, self.b), 
                             t_eval=self.domain.cpu().numpy())
        
        return solution.y
    
    def plot(self, predictions, name, title=None, size=(5, 3), save_dir=None, show=False):
        x = predictions[:,0].flatten()
        y = predictions[:,1].flatten()
        z = predictions[:,2].flatten()
        
        fig = plt.figure(figsize=size)
        
        if predictions is not None:
            plt.plot(self.domain, x, label='X(t)', linestyle='dashed')
            plt.plot(self.domain, y, label='Y(t)', linestyle='dashed')
            plt.plot(self.domain, z, label='Z(t)', linestyle='dashed')
        plt.plot(self.domain, self.solution[0], label='x(t)')
        plt.plot(self.domain, self.solution[1], label='y(t)')
        plt.plot(self.domain, self.solution[2], label='z(t)')
        
        plt.legend()
        
        if show:
            plt.show()
            
        return fig
    
    @staticmethod
    def get_problem(N):
        match N:
            case 1:
                T = 1
                s, p, b = 10, 28, 8/3
                x_0, y_0, z_0 = 0, 1, 1.05
                
            case 2:
                T = 2
                s, p, b = 10, 28, 8/3
                x_0, y_0, z_0 = 5, 1, 6
                
            case 3:
                T = 2
                s, p, b = 2, 12, 4
                x_0, y_0, z_0 = 1, 1, 1
                
            case 4:
                T = 2
                s, p, b = 5, 2, 1
                x_0, y_0, z_0 = 3, 10, 2
        
        problem = LorenzSystem(T, (s, p, b), [x_0, y_0, z_0])
        
        return problem