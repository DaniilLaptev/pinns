
import torch
import numpy as np

class FancyTensor:
    def __init__(self, x, requires_grad = True, names = None):
        
        if isinstance(x, torch.Tensor):
            self.data = x
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            self.data = torch.tensor(
                x, requires_grad=requires_grad, dtype=torch.float32
                )
        else:
            raise ValueError(f'Type {type(x)} is not supported.')
        
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
        
        self.names = names
        
    def __getitem__(self, key):
        
        if isinstance(key, str):
            if self.names is None:
                raise ValueError('Names are not provided for this points.')
            if key not in self.names:
                raise AttributeError(f'There are no such variable: {key}')
            return self.data[:, self.names.index(key)].view(-1, 1)
        
        elif isinstance(key, int):
            return self.data[key].view(1, -1)
        
        else:    
            return self.data[key].view(-1, 1)
        
    def __repr__(self):
        
        if self.names is None:
            names = str(None) + ','
        else:
            names = ''.join([name + ', ' for name in self.names])[:-2]
        
        data = '\t' + self.data.__repr__().replace('\n', '\n\t')
        string = f'Variables names: \n\t{names} \nData: \n{data}'
        return string
    
    def save(self, path):
        np.savez(path, data = self.data.detach(), names = self.names)
    
    def load(self, path, join_names = False):
        pts = np.load(path)
        
        self.data = torch.tensor(pts['data'], requires_grad=self.data.requires_grad)
        
        if join_names:
            self.names = ''.join(pts['names'].tolist())
        else:
            self.names = pts['names'].tolist()