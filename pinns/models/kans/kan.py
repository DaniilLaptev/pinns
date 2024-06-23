
from ..model import PINN

import kan
from .efficientkan import EfficientKAN
from .fastkan import FastKAN
from .fasterkan import FasterKAN
from .rbfkan import RBFKAN
from .fourierkan import FourierKAN
from .chebykan import ChebyKAN

LIBRARY = {
    'bsplines': kan.KAN,
    'efficient': EfficientKAN,
    'fast': FastKAN,
    'faster': FasterKAN,
    'rbf': RBFKAN,
    'fourier': FourierKAN,
    'chebyshev': ChebyKAN,
}

class KAN(PINN):
    def __init__(self, layers, type = 'bsplines', scale = 1, input_names = None, output_names = None, **parameters):
        super(KAN, self).__init__(input_names = input_names, output_names = output_names)
        if type == 'chebyshev':
            self.model = LIBRARY[type](layers, scale, **parameters)
        else:
            self.model = LIBRARY[type](layers, **parameters)
        
    def initialize_from_another_model(self, new_model, pts):
        
        if isinstance(new_model, PINN):
            self.model = self.model.initialize_from_another_model(new_model.model, pts)
        else:
            self.model = self.model.initialize_from_another_model(new_model, pts)
        
        return self