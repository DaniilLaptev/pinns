
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
from scipy.signal import convolve2d

from modules.utils import l2

from multiprocessing import Pool

import imageio
from tqdm import tqdm

class Problem:
    def __init__(self):
        pass
    def boundary_loss(self):
        pass
    def inner_loss(self):
        pass
    def _solve(self):
        pass

class DampedHarmonicOscillator:
    def __init__(self, T, params, initial_conditions):
        
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
    
    def plot(self, predictions, name, title=None, size=(5, 3), save_dir=None):
        fig = plt.figure(figsize=size)
        
        if predictions is not None:
            plt.plot(self.domain, predictions.flatten().cpu(), label='Model', linestyle='dashed')
        plt.plot(self.domain, self.solution, label='Numerical')
        
        plt.legend()
        if title is not None:
            plt.title(title)
        
        if save_dir is not None:
            if save_dir[-1] == '/': save_dir = save_dir[:-1]
            plt.savefig(f'{save_dir}/{name}.jpg', dpi=250)
            plt.close()
        
        else:
            plt.show()
    
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

class LotkaVolterra:
    def __init__(self, T, params, initial_conditions):
        
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
    
    def plot(self, predictions, name, title=None, size=(5, 3), save_dir=None):
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
        if title is not None:
            plt.title(title)
        
        if save_dir is not None:
            fig.set_size_inches(size)
            if save_dir[-1] == '/': save_dir = save_dir[:-1]
            plt.savefig(f'{save_dir}/{name}.jpg', dpi=250)
            plt.close()
        
        else:
            plt.show()
    
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
    
    def plot(self, predictions, name, title=None, size=(5, 3), save_dir=None):
        fig = plt.figure(figsize=size)
        
        X = self.domain[:, 0].reshape(self.Nx, self.Nt).T
        T = self.domain[:, 1].reshape(self.Nx, self.Nt).T
        ax = fig.add_subplot(111, projection='3d')
        
        if predictions is None:
            ax.plot_surface(X, T, self.solution, cmap='viridis')
        else:
            ax.plot_surface(X, T, predictions.reshape(self.Nx, self.Nt).T.numpy(), cmap='viridis')
        
        if title is not None:
            plt.title(title)
        
        if save_dir is not None:
            fig.set_size_inches(size)
            if save_dir[-1] == '/': save_dir = save_dir[:-1]
            plt.savefig(f'{save_dir}/{name}.jpg', dpi=250)
            plt.close()
        
        else:
            plt.show()
    
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
    

class GrayScott:
    def __init__(self, T, params, initial_values, dt=1.0):
        self.T = T
        self.f, self.k, self.ra, self.rb = params
        self.A_init, self.B_init = initial_values
        self.Nx, self.Ny = self.A_init.shape
        
        self.dt = dt
        
        self.solution = self._solve()
    
    @staticmethod
    def load_initial_values(matrix, points, shapes):
        for pt, shape in zip(points, shapes):
            x, y = pt
            dx, dy = shape
            matrix[x - dx : x + dx, y - dy : y + dy] = 1
        return matrix
        
    def _solve(self):
        
        A = torch.zeros((self.T, *self.A_init.shape))
        B = A.clone()
        A[0], B[0] = self.A_init, self.B_init
        
        filter = np.array([
            [0.05, 0.2, 0.05], 
            [0.2,  -1,  0.2], 
            [0.05, 0.2, 0.05]
            ])
        
        def nabla(matrix):
            conv = convolve2d(matrix, filter, mode='same')
            return torch.tensor(conv)
        
        print('Solving...')
        for i in tqdm(range(0, self.T - 1)):
            A[i+1] = A[i] + (self.ra * nabla(A[i]) - A[i]*B[i]*B[i] + self.f*(1 - A[i])) * self.dt
            B[i+1] = B[i] + (self.rb * nabla(B[i]) + A[i]*B[i]*B[i] - (self.f + self.k)*B[i]) * self.dt

        # return A[:: self.T // 100], B[:: self.T // 100]
        return A, B
    
    def get_problem(N):
    
        T = 10000
        dt = 1.0 # total time = T * dt
        Nx, Ny = 75, 75
        
        A_init = torch.ones((Nx, Ny))
        B_init = torch.zeros((Nx, Ny))
        points = [(10, 5)]
        shapes = [(3, 3)]*len(points)

        B_init = GrayScott.load_initial_values(B_init, points, shapes)

        if N == 1: 
            f, k = 0.0367, 0.0649 # mitosis
        elif N == 2: 
            f, k = 0.0545, 0.062 # coral
        else: 
            raise ValueError(f'Invalid number of problem: {N}.')
        
        ra, rb = 1.0, 0.5
        problem = GrayScott(
                T, (f, k, ra, rb), (A_init, B_init), dt
            )
        return problem
    
    def save_animation(self, path, element=1, size=(5, 3), step=1, interval=1, fps=60):
        
        fig = plt.figure(figsize=size)
        frame = plt.imshow(
            self.solution[element][0],
            origin='lower', aspect='auto', cmap='Blues', 
            extent=[0, self.Nx, 0, self.Ny],
            interpolation='gaussian'
            )
        
        def animate(i):
            matrix = self.solution[element][i]
            frame.set_data(matrix)
            frame.set_clim(vmin=0, vmax=matrix.max())
            plt.xlabel('x')
            plt.ylabel('t')
            idx = i * self.T // len(self.solution[element])
            plt.title(f'Step {idx}')
            return frame,
        
        myAnimation = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, self.T, step), \
            interval=interval, blit=True, repeat=True
            )

        myAnimation.save(path, writer=animation.FFMpegFileWriter(fps=fps))