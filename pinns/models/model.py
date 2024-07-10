import torch

class PINN:
    def __init__(self, model = None,):
        self.model = model
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            predictions = self.predict(x)
        elif isinstance(x, (list, tuple)):
            predictions = [self.predict(y) for y in x]
        elif isinstance(x, dict):
            predictions = self.predict(torch.hstack(list(x.values())))
            
        return predictions
    
    def predict(self, x):
        return self.model(x)

    def count_parameters(self):
        if isinstance(self.model, torch.nn.Module):    
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            return {'total': total, 'trainable': trainable}
        else:
            raise NotImplementedError('Parameters counter is not implemented for this model.')
    
    def init(self, weight_rule, bias_rule=torch.nn.init.zeros_, weight_args=None, bias_args=None):
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                if weight_args is not None:
                    weight_rule(param.data, *weight_args)
                else:
                    weight_rule(param.data)
            elif 'bias' in name and len(param.shape) == 1:
                if bias_args is not None:
                    bias_rule(param.data, *bias_args)
                else:
                    bias_rule(param.data)
                    
    def get_norms(self, params = True, gradients = True):
        if isinstance(self.model, torch.nn.Module):    
            norms = {}
            for name, p in self.model.named_parameters():
                if params:
                    param = p.detach().norm(p=2).item()
                if gradients:
                    gradient = p.grad.detach().norm(p=2).item() if p.grad is not None else None
                norms[name] = [param, gradient]
            return norms
        else:
            raise NotImplementedError('This method is not implemented for this model.')
    
    def parameters(self):
        if isinstance(self.model, torch.nn.Module):    
            return self.model.parameters()
        else:
            raise NotImplementedError('This method is not implemented for this model.')
    
    def get_parameters_vector(self):
        if isinstance(self.model, torch.nn.Module):    
            params, shapes = [], []
            for param in self.model.parameters():
                params.extend(param.flatten().detach())
                shapes.append(param.shape)
            params = torch.tensor(params)
        else:
            raise NotImplementedError('This method is not implemented for this model.')
        return params, shapes
        
    def __repr__(self):
        
        module = 'PyTorch' if isinstance(self.model, torch.nn.Module) else 'Unknown'
        
        return module + ' module:\n' + self.model.__repr__()