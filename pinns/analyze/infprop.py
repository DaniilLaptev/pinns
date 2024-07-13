
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
        group_at_finish = False,
        first_initial = True
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
        self.first_initial = first_initial
        
        self.precompute(domain)
        
    def _from_rectangular(self, domain):
        
        if isinstance(self.pts, dict):
            pts = torch.hstack(list(self.pts.values()))
        else:
            pts = self.pts
        
        distances = []
        if self.first_initial:
            distances.append(pts[:,0])
            start = 1
            
        else:
            start = 0
            
        for i in range(start, domain.num_vars):
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
        
    def pplot(
        self, 
        model, 
        num_points = 100, 
        R = 1, 
        loss_vmax = None, 
        descending = False, 
        figsize = (10, 5), 
        interpolation = None
        ):
        
        p, _ = model.get_parameters_vector()
        
        rads = torch.linspace(0, R, num_points)
        result = {'losses': [], 'errors': []}

        for i in range(num_points):
            points = torch.randn(p.size(0))
            points /= torch.norm(points, p = 2)
            random_parameters = p + rads[i] * points
            
            start = 0
            for param in model.parameters():
                step = param.shape[0] * param.shape[1] if len(param.shape) > 1 else param.shape[0]
                end = start + step
                param.data = random_parameters[start:end].reshape(param.shape)
                start = end
                
            preds = model(self.pts)
            ls = self.residual(preds, self.pts).abs().flatten()
            result['losses'].append(ls.detach())
            
            if self.vals is not None:
                er = (preds - self.vals).abs().mean(axis=1)
                result['errors'].append(er.detach())
        
        if descending:
            rads = rads[::-1]
        result['losses'] = torch.vstack(result['losses']).T
        if self.vals is not None:
            result['errors'] = torch.vstack(result['errors']).T
        
        extent = [rads[0], rads[-1], 0, self.dists.max()]
        kwargs = {
            'aspect': 'auto',
            'origin': 'lower',
            'interpolation': interpolation,
            'extent': extent
            }
        
        def plot_loss(ax1, ax2):
            im = ax1.imshow(result['losses'], vmax = loss_vmax, **kwargs)
            plt.colorbar(im)
            ax1.title.set_text('Loss')
            ax1.set_ylabel('Distance')
            ax2.plot(rads, result['losses'].mean(dim=0))
            ax2.set_ylabel('Mean Loss')
        
        def plot_error(ax1, ax2):
            im = ax1.imshow(result['errors'], **kwargs)
            plt.colorbar(im)
            ax1.title.set_text('Error')
            ax1.set_ylabel('Distance')
            ax2.plot(rads, result['errors'].mean(dim=0))
            ax2.set_ylabel('Mean Error')
        
        if self.strategy == 'both':
            fig, axs = plt.subplots(2, 2, figsize=figsize, width_ratios=[5, 2])
            plot_loss(axs[0][0], axs[0][1])
            plot_error(axs[1][0], axs[1][1])
            axs[1][0].set_xlabel('Radius')
            axs[1][1].set_xlabel('Radius')
            for ax in [axs[0][1], axs[1][1]]:
                ax.set_xlim(rads[0], rads[-1])
                ax.set_yscale('log')
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            if self.strategy == 'loss':
                plot_loss(axs[0], axs[1])
            else:
                plot_error(axs[0], axs[1])
            axs[0].set_xlabel('Radius')
            axs[1].set_xlabel('Radius')
            axs[1].set_xlim(rads[0], rads[-1])
            axs[1].set_yscale('log')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            
        plt.tight_layout()
        plt.show()
    
    def describe(self):
        pass
    