import torch

from pinns.metrics import l2

from tqdm.notebook import tqdm_notebook as tqdm

from collections import defaultdict
import matplotlib.pyplot as plt

from pinns.models import PINN
from pinns.samplers import Sampler

class Trainer:
    def __init__(
        self, 
        lossfn : callable,
        model : PINN,
        constraints_sampler : Sampler,
        collocation_sampler : Sampler,
        loss_coefs : list = None,
        coef_adjuster = None,
        analyzers : list = None,
        ):
        
        self.model = model
        self.optimizer = None
        
        self.cstr_sampler = constraints_sampler
        self.cllc_sampler = collocation_sampler
        self.loss_coefs = loss_coefs
        self.cstr = defaultdict()
        self.cllc = defaultdict()
        
        self.lossfn = lossfn
        self.iter = 0
        
        self.coef_adjuster = coef_adjuster
        
        self.loss_history = []
        self.error_history = []
        
        self.analyzers = analyzers
        
    def evaluate(self, error_metric, pts, vals):
        pred = self.model(pts)
        error = error_metric(pred, vals)
        return error
        
    def train_iter(self):
        
        def closure():
            
            self.optimizer.clear_cache()
            
            self.cstr['pred'] = self.model(self.cstr['pts'])
            self.cllc['pred'] = self.model(self.cllc['pts'])
            
            current_losses = self.lossfn(
                self.cstr['pts'],
                self.cstr['pred'],
                self.cstr['vals'],
                self.cllc['pts'],
                self.cllc['pred']
            )
            
            total_loss = 0
            for loss, w in zip(current_losses, self.loss_coefs):
                total_loss += loss * w
            
            total_loss.backward(retain_graph=True)
            
            return total_loss
        
        if self.optimizer.mode == 'closure':
            loss = self.optimizer.step(closure)
            
        else:
            loss = closure()
            self.optimizer.step()

        self.loss_history.append(loss.item())
            
        return loss
    
    def train(
        self, 
        num_iters : int,
        optimizers : list,
        show_progress : bool = True,
        validate_every : int = None,
        test_sampler : Sampler = None,
        metrics : list = [l2],
        idx_printed_metric : list = None,
        training_start_callbacks : list = [],
        epoch_end_callbacks : list = [],
        training_end_callbacks : list = []
        ):
        
        iters = []
        optims = []
        for (iter, optim) in optimizers:
            iters.append(iter)
            optims.append(optim)
        current_optim = 0
        iters.append(num_iters + 1)
        self.optimizer = optims[0]
        
        if validate_every is not None:
            if test_sampler is None:
                raise ValueError('Test sampler must be provided for validation.')
            
            # self.model.eval()
            with torch.no_grad():
                pts, vals = test_sampler()
                metric_results = [self.evaluate(metric, pts, vals) for metric in metrics]
            self.error_history.append(torch.tensor(metric_results).view(-1, 1))
            # self.model.train()
            
        if show_progress:
            pbar = tqdm(range(num_iters))
            if isinstance(idx_printed_metric, int):
                idx_printed_metric = [idx_printed_metric]
            if idx_printed_metric is None:
                idx_printed_metric = range(len(metrics))
            
        for callback in training_start_callbacks:
            callback()
            
        if self.analyzers is not None:
            for analyzer in self.analyzers:
                analyzer.execute(self.model)
            
        for i in range(num_iters):
            
            self.cllc['pts'] = self.cllc_sampler()
            self.cstr['pts'], self.cstr['vals'] = self.cstr_sampler()
            
            loss = self.train_iter()
            
            if torch.isnan(loss):
                print(f'Loss became NaN at iteration {i}. Training stops.')
                break
            
            if torch.isinf(loss):
                print(f'Loss became inf at iteration {i}. Training stops.')
                break
            
            if self.coef_adjuster is not None:
                self.loss_coefs = self.coef_adjuster()
                
            if self.optimizer.scheduler is not None:
                self.optimizer.scheduler.step()
                
            if i == iters[current_optim + 1]:
                self.optimizer = optims[current_optim + 1]
                current_optim += 1
                
            if validate_every is not None and (i + 1) % validate_every == 0:
                with torch.no_grad():
                    pts, vals = test_sampler()
                    metric_results = [self.evaluate(metric, pts, vals) for metric in metrics]
                self.error_history.append(torch.tensor(metric_results).view(-1, 1))
                
            if show_progress:
                desc = f'Loss: {loss:.5f}'
                if validate_every is not None:
                    desc += ', metrics: '
                    printed_metrics = [metric_results[idx] for idx in idx_printed_metric]
                    for metric_result in printed_metrics:
                        desc += f'{metric_result:.5f}, '
                    desc = desc[:-2]
                pbar.set_description(desc)
                pbar.update(1)
            self.iter += 1
            
            for callback in epoch_end_callbacks:
                callback()
                
            if self.analyzers is not None:
                for analyzer in self.analyzers:
                    if i % analyzer.compute_every == 0:
                        analyzer.execute(self.model)
                
        for callback in training_end_callbacks:
                callback()
                
        if self.analyzers is not None:
            for analyzer in self.analyzers:
                analyzer.finish()
                
    def plot(
        self,
        error_names = None,
        figsize = (10, 4),
        single = True,
        stack = 'horizontal', 
        width_ratios = [1, 1], 
        height_ratios = [1, 1],
        loss_log = True,
        error_log = True,
        ):
        
        loss_history = torch.tensor(self.loss_history)
        
        plot_error = len(self.error_history) > 0
        
        if plot_error:
            error_history = torch.hstack(self.error_history)
            if error_names is None:
                error_names = [f'Metric {i}' for i in range(error_history.shape[0])]
        else:
            single = True    
        
        if single:
            
            fig = plt.figure(figsize=figsize)
            plt.plot(loss_history, label = 'Loss')
            
            if plot_error:
                for er, name in zip(error_history, error_names):
                    plt.plot(er, label = name, linestyle = ':')
            plt.legend()
            plt.grid()
            
            if error_log or loss_log:
                plt.yscale('log')
            
        else:
            
            if stack == 'horizontal':
                fig, axs = plt.subplots(1, 2, figsize=figsize, width_ratios=width_ratios, sharex=True)
            else:
                fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=height_ratios, sharex=True)
            
            axs[0].plot(loss_history, label = 'Loss')
            axs[0].legend()
            axs[0].grid()
            if loss_log:
                axs[0].set_yscale('log')
            
            for er, name in zip(error_history, error_names):
                axs[1].plot(er, label = name)
            axs[1].legend()
            axs[1].grid()
            if error_log:
                axs[1].set_yscale('log')
        
        plt.show()