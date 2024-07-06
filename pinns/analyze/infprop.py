
import torch
import numpy as np
import matplotlib.pyplot as plt

class InformationPropagation:
    def __init__(
        self,
        points,
        domain,
        values = None,
        strategy = 'loss',
        compute_every = 1,
        residual = None,
        group_at_finish = False
        ):
        
        if strategy in ['error', 'both'] and values is None:
            raise ValueError('For strategy "error" you must provide true values.')
        if strategy in ['loss', 'both'] and residual is None:
            raise ValueError('For strategy "loss" you must provide residual loss function.')
        
        self.pts = points
        self.vals = values
        
        self.fn = None
        self.strategy = strategy
        self.compute_every = compute_every
        self.total_computed = 0
        
        self.residual = residual
        
        self.loss_history = []
        self.error_history = []
        self.dists = None
        
        self.group_at_finish = group_at_finish
        
        self.precompute(domain)
        
    def _from_rectangular(self, domain):
        
        if isinstance(self.pts, dict):
            pts = torch.hstack(list(self.pts.values()))
        else:
            pts = self.pts
        distances = [pts[:,0]]
        
        for i in range(1, domain.num_vars):
            A, B = domain[i]
            left_dists  = (pts[:,i] - A).abs()
            right_dists = (pts[:,i] - B).abs()
            distances.append(left_dists)
            distances.append(right_dists)
        
        distances = torch.stack(distances).detach().T.min(dim=1).values
        self.dists, idx = distances.sort(dim=-1)
        
        if isinstance(self.pts, dict):
            for k, v in self.pts.items():
                self.pts[k] = v[idx]
        else:
            self.pts = self.pts[idx]
        self.vals = self.vals[idx]
        
        fnlib = {
            "both":  self._both,
            "loss":  self._loss,
            "error": self._error,
        }
        self.fn = fnlib[self.strategy]
    
    def precompute(self, domain):
        """Precompute distances and history.
        
        Can only handle 1D Dirichlet boundary conditions (for now).
        We assume that time coordinate are going first.

        Parameters
        ----------
        trainer : Trainer
            Trainer instance to get data from.
        """
        
        if domain.shape == 'rectangular':
            self._from_rectangular(domain)
            
        else:
            raise NotImplementedError(f'Calculation for domain shape {domain.shape} is not implemented.')
    
    def _loss(self, preds):
        loss = self.residual(preds, self.pts).abs().flatten()
        self.loss_history.append(loss.detach())
        
    def _error(self, preds):
        error = (preds - self.vals).abs().mean(axis=1)
        self.error_history.append(error.detach())
        
    def _both(self, preds):
        self._loss(preds)
        self._error(preds)
    
    def execute(self, model):
        preds = model(self.pts)
        self.fn(preds)
        self.total_computed += 1
        
    def finish(self):
        if self.strategy in ['loss', 'both']:
            self.loss_history = torch.vstack(self.loss_history).T
        if self.strategy in ['error', 'both']:
            self.error_history = torch.vstack(self.error_history).T
        if self.group_at_finish:
            self.loss_history, self.error_history = self.group()
    
    def group(self):
        unique, amount = torch.unique(self.dists, return_counts=True)
        grouped_losses = []
        grouped_errors = []
        start = 0
        for i in range(len(amount)):
            end = start + amount[i]
            group = np.arange(start, end)
            if self.strategy in ['loss', 'both']:
                grouped_losses.append(self.loss_history[group].mean(axis=0).squeeze(0))
            if self.strategy in ['error', 'both']:
                grouped_errors.append(self.error_history[group].mean(axis=0).squeeze(0))
            start = end
            
        if self.strategy in ['loss', 'both']:
            grouped_losses = torch.vstack(grouped_losses)
        if self.strategy in ['error', 'both']:
            grouped_errors = torch.vstack(grouped_errors)
            
        return grouped_losses, grouped_errors
    
    def plot(self, figsize = (10, 5), grouped = False, iterations = None, interpolation = None):
        
        if iterations is None:
            iterations = self.total_computed
        
        if grouped:
            if self.group_at_finish:
                print('Arrays are already grouped.')
                grouped = False
            else:
                grouped_losses, grouped_errors = self.group()
            
        kwargs = {
            'aspect': 'auto',
            'origin': 'lower',
            'interpolation': interpolation,
            'extent':[0, iterations, 0, self.dists.max()]
            }
        
        def plot_loss(ax):
            if grouped:
                im = ax.imshow(grouped_losses, **kwargs)
            else:
                im = ax.imshow(self.loss_history, **kwargs)
            plt.colorbar(im)
            ax.title.set_text('Loss')
            ax.set_ylabel('Distance')
        
        def plot_error(ax):
            if grouped:
                im = ax.imshow(grouped_errors, **kwargs)
            else:
                im = ax.imshow(self.error_history, **kwargs)
            plt.colorbar(im)
            ax.title.set_text('Error')
            ax.set_ylabel('Distance')
        
        if self.strategy == 'both':
            fig, axs = plt.subplots(2, 1, figsize=figsize)
            plot_loss(axs[0])
            plot_error(axs[1])
            axs[1].set_xlabel('Iterations')
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            if self.strategy == 'loss':
                plot_loss(ax)
            else:
                plot_error(ax)
            ax.set_xlabel('Iterations')
            
        plt.tight_layout()
        plt.show()
    
    def describe(self):
        pass
    