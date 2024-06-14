import torch

from pinns.errors import l2_error

from tqdm.notebook import tqdm_notebook as tqdm

from collections import defaultdict

class Trainer:
    def __init__(
        self, 
        loss_fn, 
        model, 
        constraints_sampler,
        collocation_sampler,
        loss_coefs,
        logger = None,
        coef_adjuster = None,
        test_points_sampler = None,
        ):
        
        self.model = model
        self.optimizer = None
            
        self.constraints_sampler = constraints_sampler
        self.collocation_sampler = collocation_sampler
        self.collocation = defaultdict()
        self.constraints = defaultdict()
        
        self.loss_fn = loss_fn
        self.loss_coefs = loss_coefs
        self.loss_history = []
        self.iter = 0
        
        self.error_history = []
        self.test_points = defaultdict()
        self.test_points_sampler = test_points_sampler
        
        self.coef_adjuster = coef_adjuster
        
    def evaluate(self, error_metric, **sampler_args):
        self.test_points['pts'], self.test_points['vals'] = self.test_points_sampler(**sampler_args)
        self.test_points['pred'] = self.model.predict(self.test_points['pts'])
        error = error_metric(self.test_points['pred'], self.test_points['vals'])
        return error
        
    def train_iter(self):
        
        def closure():
            self.optimizer.clear_cache()
            
            self.constraints['pred'] = self.model.predict(self.constraints['pts'])
            self.collocation['pred'] = self.model.predict(self.collocation['pts'])
            
            losses = self.loss_fn(
                self.constraints['pts'],
                self.constraints['pred'],
                self.constraints['vals'],
                self.collocation['pts'],
                self.collocation['pred']
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
            
        return loss
    
    def train(
        self, 
        num_iters, 
        optimizers,
        show_progress = True,
        validate_every = None,
        error_metric = None,
        checkpoint_strategy = None,
        checkpoint_name = None,
        checkpoint_every = None,
        ):
        
        iters = []
        optims = []
        for (iter, optim) in optimizers:
            iters.append(iter)
            optims.append(optim)
        current_optim = 0
        iters.append(num_iters + 1)
        
        self.optimizer = optims[current_optim]
        
        if error_metric is None:
            error_metric = l2_error
        
        # todo: measure not only best error, but best loss (in case we dont have solution)
        if validate_every is not None:
            if self.test_points_sampler is None:
                raise ValueError('Test points sampler must be provided for validation.')
            error = self.evaluate(error_metric)
            self.error_history.append(error)
        
        if checkpoint_every is not None:
            if checkpoint_strategy == 'best':
                best_result = self.error_history[-1]
            else:
                raise NotImplementedError('This saving method is not implemented yet.')
            
        if show_progress:
            pbar = tqdm(range(num_iters))
            
        for i in range(num_iters):
            
            self.collocation['pts'] = self.collocation_sampler()
            
            batch = self.constraints_sampler()
            self.constraints['pts'], self.constraints['vals'] = batch
            
            loss = self.train_iter()
            
            if torch.isnan(loss):
                print(f'Loss became NaN at iteration {i}. Training stops.')
                break
            
            if torch.isinf(loss):
                print(f'Loss became inf at iteration {i}. Training stops.')
                break
                
            self.loss_history.append(loss.item())
            
            if show_progress:
                pbar.set_description(f'Loss: {loss:.5f}')
                pbar.update(1)
            self.iter += 1
            
            if self.coef_adjuster is not None:
                self.loss_coefs = self.coef_adjuster(self.iter, self.loss_coefs)
                
            if self.optimizer.scheduler is not None:
                self.optimizer.scheduler.step()
                
            if i == iters[current_optim + 1]:
                self.optimizer = optims[current_optim + 1]
                current_optim += 1
                
            if validate_every is not None and (i + 1) % validate_every == 0:
                error = self.evaluate(error_metric)
                self.error_history.append(error)
                
            if checkpoint_strategy == 'best' and error < best_result:
                torch.save(self.model.model, checkpoint_name + '.pt')
                best_result = loss.item()
            
            elif checkpoint_strategy == 'every' and i % checkpoint_every == 0:
                torch.save(self.model.model, checkpoint_name + f'_{i}.pt')