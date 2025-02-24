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

    def predict(self, x):
        n = len(x)
        x_b = np.c_[np.ones((n,1)),x]
        y_pred = x_b.dot(self.theta)
        return y_pred

class DGRegresion:
    def __init__(self, n_iter = 1000, mini_lote = None, semilla = None, eta0 = None):
        self.mini_lote = mini_lote
        self.semilla = semilla
        self.eta0 = eta0
        self.n_iter = n_iter
        self.theta = None
        self.intercept = None
        self.coef = None

    def aprendizaje(self,epoch,eta0):
        return(eta0/(1+epoch))

    def fit(self,x,y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)),x]
        self.theta = np.random.rand(n+1,1)
        if(self.mini_lote is None and self.semilla is None):
            for i in range(self.n_iter):
                eta = self.aprendizaje(i,self.eta0)
                grad = (2/m)*x_b.T.dot(x_b.dot(self.theta)-y)
                self.theta = self.theta-eta*grad
                self.intercept = self.theta[0]
                self.coef = self.theta[1:]
        elif(self.mini_lote is not None and self.semilla is None):
            for i in range(self.n_iter):
                index_m = np.random.permutation(m)
                x_bm = x_b[index_m]
                y_m = y[index_m]
                for j in range(0,m,self.mini_lote):
                    eta = self.aprendizaje(i,self.eta0)
                    xi = x_bm[i:i+self.mini_lote]
                    yi = y_m[i:i+self.mini_lote]
                    grad = (2/self.mini_lote)*xi.T.dot(xi.dot(self.theta)-yi)
                    self.theta = self.theta-eta*grad
                    self.intercept = self.theta[0]
                    self.coef = self.theta[1:]
        elif(self.mini_lote is None and self.semilla is not None):
            for i in range(self.n_iter):
                for j in range(m):
                    np.random.seed(self.semilla)
                    random_index = np.random.randint(m)
                    xi = x_b[random_index:random_index+1]
                    yi = y[random_index:random_index+1]
                    eta = self.aprendizaje(i,self.eta0)
                    grad = (2)*xi.T.dot(xi.dot(self.theta)-yi)
                    self.theta = self.theta-eta*grad
                    self.intercept = self.theta[0]
                    self.coef = self.theta[1:]


                