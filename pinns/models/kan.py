
import torch
import torch.nn as nn

from .model import PINN

import kan

class KAN(PINN):
    def __init__(self, layers, parameters = None):
        
        self.model = kan.KAN(layers, **parameters)