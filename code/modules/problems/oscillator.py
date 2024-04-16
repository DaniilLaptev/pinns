
import torch
from torch.autograd import grad

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from modules.utils import l2

from modules.problems.problem import Problem

class DampedHarmonicOscillator(Problem):
    def __init__(self, T, params, initial_conditions):
        super(DampedHarmonicOscillator, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.T = T
        self.d, self.w0 = params
        self.init_vals = torch.tensor(initial_conditions).to(self.device)

        self.domain = torch.linspace(0, self.T, 128).to(self.device)
        self.solution = self._solve()
        
    def test(self, model):
        return model(self.domain.reshape(-1, 1)).detach().cpu()
        
    def error(self, predictions):
        return float(l2(predictions.flatten(), self.solution))
    
    def boundary_loss(self, model, t, lbcoefs):
        x = model(t)
        v = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        loss = torch.mean(torch.square(torch.hstack([x, v]) - self.init_vals))
        return lbcoefs[0] * loss
    
    def inner_loss(self, model, t, lfcoefs):
        x = model(t)
        v = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        a = grad(v, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        loss = torch.mean(torch.square(a + 2*self.d*self.w0 * v + self.w0**2 * x))
        return lfcoefs[0] * loss
    
    def _solve(self):
        def harmonic_oscillator(t, y, d, w0):
            x, v = y
            dxdt = v
            dvdt = -2*d*w0 * v - w0**2 * x
            return [dxdt, dvdt]

        solution = solve_ivp(harmonic_oscillator, 
                             (0, self.T), 
                             self.init_vals.cpu().numpy(), 
                             args=(self.d, self.w0), 
                             t_eval=self.domain.cpu().numpy())
        return solution.y[0]
    
    def plot(self, predictions, name, title=None, size=(5, 3), show=False):
        fig = plt.figure(figsize=size)
        
        if predictions is not None:
            plt.plot(self.domain, predictions.flatten().cpu(), label='Model', linestyle='dashed')
        plt.plot(self.domain, self.solution, label='Numerical')
        
        plt.legend()
        
        if show:
            plt.show()
        return fig
            
    @staticmethod        
    def get_problem(N):
        match N:
            case 1:
                T = 10
                zeta, omega = 0.2, 2.0
                x_0, v_0 = 5.0, 7.0
            case 2:
                T = 10
                zeta, omega = 0.7, 1.0
                x_0, v_0 = 2.0, 3.0
            case 3:
                T = 100
                zeta, omega = 0.9, 0.1
                x_0, v_0 = 1.0, -5.0
            case 4:
                T = 5
                zeta, omega = 0.1, 8
                x_0, v_0 = -3.0, 10.
        
        problem = DampedHarmonicOscillator(T, (zeta, omega), (x_0, v_0))
        
        return problem