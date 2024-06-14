
from .model import PINN

import kan

class KAN(PINN):
    def __init__(self, layers, **parameters):
        
        self.model = kan.KAN(layers, **parameters)
        
    def initialize_from_another_model(self, new_model, pts):
        
        if isinstance(new_model, PINN):
            self.model = self.model.initialize_from_another_model(new_model.model, pts)
        else:
            self.model = self.model.initialize_from_another_model(new_model, pts)
        
        return self