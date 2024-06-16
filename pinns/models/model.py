import torch

class PINN:
    def __init__(self, pytorch_module):
        self.model = pytorch_module
    
    def __call__(self, x):
        return self.predict(x)
    
    def predict(self, x):
        if isinstance(x, (list, tuple)):
            return [self.model(pts) for pts in x]
        if isinstance(x, dict):
            pts = torch.hstack(list(x.values()))
            return self.model(pts)
        return self.model(x)
    
    def count_parameters(self):
        total = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad: continue
            total += p.numel()
        return total
    
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
                    
    def get_norms(model):
        norms = {}
        for name, param in model.model.named_parameters():
            weight = param.detach().norm(p=2).item()
            gradient = param.grad.detach().norm(p=2).item() if param.grad is not None else None
            norms[name] = [float(weight), float(gradient)]
        return norms
    
    def get_params(model):
        params = []
        for param in model.parameters():
            params.extend(param.flatten().detach())
        return torch.tensor(params)