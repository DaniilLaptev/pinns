import torch

from pinns.metrics import l2

from tqdm.notebook import tqdm_notebook as tqdm

from collections import defaultdict

class Trainer:
    def __init__(
        self, 
        lossfn,
        model,
        constraints_sampler,
        collocation_sampler,
        loss_coefs = None,
        coef_adjuster = None,
        log_every_loss = False
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
        
        self.log_every_loss = log_every_loss
        self.loss_history = []
        self.error_history = []
        
    def evaluate(self, error_metric, pts, vals):
        pred = self.model(pts)
        error = error_metric(pred, vals)
        return error
        
    def train_iter(self):
        
        def closure():
            self.optimizer.clear_cache()
            
            self.cstr['pred'] = self.model(self.cstr['pts'])
            self.cllc['pred'] = self.model(self.cllc['pts'])
            
            losses = self.lossfn(
                self.cstr['pts'],
                self.cstr['pred'],
                self.cstr['vals'],
                self.cllc['pts'],
                self.cllc['pred']
            )
            
            total_loss = 0
            for loss, w in zip(losses, self.loss_coefs):
                total_loss += loss * w
            
            total_loss.backward(retain_graph=True)
            
            return total_loss
        
        if self.optimizer.mode == 'closure':
            loss = self.optimizer.step(closure)
            
        else:
            loss = closure()
            self.optimizer.step()
        
        if self.log_every_loss:
            curr_losses = self.lossfn(
                    self.cstr['pts'],
                    self.cstr['pred'],
                    self.cstr['vals'],
                    self.cllc['pts'],
                    self.cllc['pred']
                )
            self.loss_history.append([ls.item() for ls in curr_losses] + [loss.item()])
        else:
            self.loss_history.append(loss.item())
            
        return loss
    
    def train(
        self, 
        num_iters,
        optimizers,
        show_progress = True,
        validate_every = None,
        test_sampler = None,
        metrics = [l2],
        idx_printed_metric = None,
        training_start_callbacks = [],
        epoch_end_callbacks = [],
        training_end_callbacks = []
        ):
        
        iters = []
        optims = []
        for (iter, optim) in optimizers:
            iters.append(iter)
            optims.append(optim)
        current_optim = 0
        iters.append(num_iters + 1)
        self.optimizer = optims[0]
        
        if metrics is None:
            metrics = [l2]
        
        if validate_every is not None:
            if test_sampler is None:
                raise ValueError('Test sampler must be provided for validation.')
            
            # self.model.eval()
            with torch.no_grad():
                pts, vals = test_sampler()
                metric_results = [self.evaluate(metric, pts, vals) for metric in metrics]
            self.error_history.append(metric_results)
            # self.model.train()
            
        if show_progress:
            pbar = tqdm(range(num_iters))
            if isinstance(idx_printed_metric, int):
                idx_printed_metric = [idx_printed_metric]
            if idx_printed_metric is None:
                idx_printed_metric = range(len(metrics))
            
        for callback in training_start_callbacks:
            callback()
            
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
                self.error_history.append(metric_results)
                
            if show_progress:
                desc = f'Loss: {loss:.5f}'
                if validate_every is not None:
                    desc += ', metrics: '
                    printed_metrics = [metric_results[idx] for idx in idx_printed_metric]
                    for metric_result in printed_metrics:
                        desc += f'{metric_result:.5f}, '
                pbar.set_description(desc[:-2])
                pbar.update(1)
            self.iter += 1
            
            for callback in epoch_end_callbacks:
                callback()
                
        for callback in training_end_callbacks:
                callback()