
import numpy as np

from torch import Tensor

def l2(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
        
    return np.sqrt(np.square(predicts - target).sum(axis=0)).mean()

def rel_l2(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
    
    distance = np.sqrt(np.square(predicts - target).sum(axis=0))
    norm = np.sqrt(np.square(target).sum(axis=0))
    
    return (distance / norm).mean()

def mse(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
    
    return np.square(predicts - target).mean(axis=0).mean()

def rmse(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
    
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
        
    return np.sqrt(np.square(predicts - target).mean(axis=0)).mean()