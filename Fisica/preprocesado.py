import numpy as np

class PL:
    
    def __init__(self, grado):
        self.grado = grado

    def fit_transform(self, x):
        n = len(x)
        x_l = np.column_stack([x**i for i in range(0, self.grado + 1)]) #Lo que se genere dentro del paréntesis se acomoda en un arreglo en forma de columna
        return x_l
        
        