
from .model import PINN

import kan

class KAN(PINN):
    def __init__(self, layers, **parameters):
        
        self.model = kan.KAN(layers, **parameters)