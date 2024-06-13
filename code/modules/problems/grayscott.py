
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
from torch.autograd import grad

from scipy.signal import convolve2d

from modules.problems.problem import Problem

class GrayScott(Problem):
    def __init__(self, T, params, initial_values, dt=1.0):
        super(GrayScott, self).__init__()
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