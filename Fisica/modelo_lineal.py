import numpy as np

class RegresionLineal: 
    def __init__(self):
        self.theta = None
        self.intercept = None
        self.coef = None
        
    def fit(self,x,y):
        n = len(x)
        x_b = np.c_[np.ones((n,1)),x]
        self.theta = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
        self.intercept = self.theta[0]
        self.coef = self.theta[1:]