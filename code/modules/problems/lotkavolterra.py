
import torch
from torch.autograd import grad

import numpy as np 

from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt

from modules.utils import l2

from modules.problems.problem import Problem

class LotkaVolterra(Problem):
    def __init__(self, T, params, initial_conditions):
        super(LotkaVolterra, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.T = T
        self.alpha, self.beta, self.delta, self.gamma = params
        self.init_vals = torch.tensor(initial_conditions).to(self.device)

        self.domain = torch.linspace(0, self.T, 128).to(self.device)
        self.solution = self._solve()
        
    def test(self, model):
        return model(self.domain.reshape(-1, 1)).detach().cpu()
        
    def error(self, predictions):
        x = predictions[:,0].flatten()
        y = predictions[:,1].flatten()
        error_x = l2(x, self.solution[0])
        error_y = l2(y, self.solution[1])
        return float((error_x + error_y) / 2)
    
    def boundary_loss(self, model, t, lbcoefs):
        x = model(t)
        loss = torch.mean(torch.square(x - self.init_vals))
        return lbcoefs[0] * loss
    
    def inner_loss(self, model, t, lfcoefs):
        xy = model(t)
        x = xy[:,[0]]
        y = xy[:,[1]]

        dX = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        dY = grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        
        loss_dX = torch.mean(torch.square(dX - self.alpha * x + self.beta * x * y))
        loss_dY = torch.mean(torch.square(dY - self.delta * x * y + self.gamma * y))
        
        return lfcoefs[0] * loss_dX + lfcoefs[1] * loss_dY
    
    def _solve(self):
        def lotka_volterra(t, u, alpha, beta, delta, gamma):
            x, y = u
            dx_dt = alpha * x - beta * x * y
            dy_dt = delta * x * y - gamma * y
            return [dx_dt, dy_dt]

        solution = solve_ivp(lotka_volterra, 
                             (0, self.T),
                             self.init_vals.cpu().numpy(), 
                             method='RK45',
                             args=(self.alpha, self.beta, self.delta, self.gamma), 
                             t_eval=self.domain.cpu().numpy())
        
        return solution.y
    
    def plot(self, predictions, name, title=None, size=(5, 3), save_dir=None):
        x = predictions[:,0].flatten()
        y = predictions[:,1].flatten()
        
        fig = plt.figure(figsize=size)
        
        if predictions is not None:
            plt.plot(self.domain, x, label='X(t)', linestyle='dashed')
            plt.plot(self.domain, y, label='Y(t)', linestyle='dashed')
        plt.plot(self.domain, self.solution[0], label='x(t)')
        plt.plot(self.domain, self.solution[1], label='y(t)')
        
        plt.legend()
        if title is not None:
            plt.title(title)
        
        if save_dir is not None:
            fig.set_size_inches(size)
            if save_dir[-1] == '/': save_dir = save_dir[:-1]
            plt.savefig(f'{save_dir}/{name}.jpg', dpi=250)
            plt.close()
        
        else:
            plt.show()
            
    def plot_frequencies(self, predictions, size=(5, 5), show=False):
        
        xsol, ysol = self.solution
        fxsol, fysol = fft(xsol - xsol.mean()), fft(ysol - ysol.mean())
        
        xpred, ypred = predictions.T
        fxpred, fypred = fft(xpred - xpred.mean()), fft(ypred - ypred.mean())
        
        N = self.domain.numpy().shape[0]
            
        fig, axs = plt.subplots(2, 1, figsize=size)
        
        axs[0].plot(self.domain.numpy(), xsol, label='x(t)')
        axs[0].plot(self.domain.numpy(), ysol, label='x(t)')
        axs[0].plot(self.domain.numpy(), xpred, label='X(t)', linestyle='dashed')
        axs[0].plot(self.domain.numpy(), ypred, label='Y(t)', linestyle='dashed')
        axs[0].legend()
        
        dt = self.domain.numpy()[1]
        
        freqs = fftfreq(len(fxsol), dt)
        axs[1].plot(freqs[:N//2], np.abs(fxsol)[:N//2], label='x(t)')
        axs[1].plot(freqs[:N//2], np.abs(fysol)[:N//2], label='y(t)')
        axs[1].plot(freqs[:N//2], np.abs(fxpred)[:N//2], label='X(t)', linestyle='dashed')
        axs[1].plot(freqs[:N//2], np.abs(fypred)[:N//2], label='Y(t)', linestyle='dashed')
        
        axs[1].set_xscale('log')
        axs[1].legend()
        
        if show:
            plt.show()
        return fig
    
    @staticmethod
    def get_problem(N):
        match N:
            case 1:
                T = 25
                alpha, beta, delta, gamma = 0.4, 0.1, 0.1, 0.6
                x_0, y_0 = 10, 5
                
            case 2:
                T = 16
                alpha, beta, delta, gamma = 0.7, 1.5, 0.3, 0.7
                x_0, y_0 = 2, 2
                
            case 3:
                T = 100
                alpha, beta, delta, gamma = 0.2, 0.2, 0.2, 0.2
                x_0, y_0 = 10, 10
                
            case 4:
                T = 38
                alpha, beta, delta, gamma = 0.4, 0.1, 0.1, 0.6
                x_0, y_0 = 5, 5
        
        problem = LotkaVolterra(T, (alpha, beta, delta, gamma), [x_0, y_0])
        
        return problem