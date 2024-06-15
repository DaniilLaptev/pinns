import torch
import numpy as np

from torch import Tensor

def l2_error(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
        
    return np.sqrt(np.square(predicts - target).sum())

def rel_l2_error(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    if predicts.shape != target.shape:
        raise RuntimeError('Shapes of predicts and targets are different.')
    
    return np.linalg.norm(predicts - target, 2) / np.linalg.norm(target, 2)

def rmse_error(predicts, target):
    
    if isinstance(predicts, Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
        
    return np.sqrt(np.square(predicts - target).mean())