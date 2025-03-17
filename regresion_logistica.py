import numpy as np

class RegresionLogistica:
    def __init__(self, epochs = 1000, eta0 = 0.1, umbral = 0.5):
        self.epochs = epochs
        self.eta0 = eta0 
        self.umbral = umbral
        self.theta = None
        self.intercept = None
        self.coef = None

    def sigmoidea(self, z):
        return (1/(1+np.exp(-z)))

    def softmax(self,z):
        exp_z = np.exp(z)
        return (exp_z/np.sum(exp_z,axis = 1, keepdims = True))

    def aprendizaje(self, eta0, epoch):
        return (eta0/(1 + epoch))

    def fit(self, x, y):
        m, n = x.shape
        num_clases = len(np.unique(y))
        if num_clases == 2:
            x_b = np.c_[np.ones((m,1)),x]
            self.theta = np.zeros((n+1,1))
            for epoch in range(self.epochs):
                for i in range(m):
                    indice_aleatorio = np.random.randint(m)
                    xi = x_b[indice_aleatorio:indice_aleatorio+1]
                    yi = y[indice_aleatorio:indice_aleatorio+1]
                    zi = np.dot(xi,self.theta)
                    hi = self.sigmoidea(zi)
                    gra = np.dot(xi.T,(hi-yi))
                    eta = self.aprendizaje(self.eta0, epoch)
                    self.theta -= eta*gra
                    self.intercept = self.theta[0,0]
                    self.coef = self.theta[1:,0]
        else:
            num_datos = len(y)
            y_one = np.zeros((num_datos, num_clases))
            for i in range(num_datos):
                y_one[i,y[i]] = 1
            
            x_b = np.c_[np.ones((m,1)),x]
            self.theta = np.zeros((n+1,num_clases))
            for epoch in range(self.epochs):
                for i in range(m):
                    indice_aleatorio = np.random.randint(m)
                    xi = x_b[indice_aleatorio:indice_aleatorio+1]
                    yi = y_one[indice_aleatorio:indice_aleatorio+1]
                    zi = np.dot(xi,self.theta)
                    hi = self.softmax(zi)
                    gra = np.dot(xi.T,(hi-yi))
                    eta = self.aprendizaje(self.eta0, epoch)
                    self.theta -= eta*gra
                    self.intercept = self.theta[0,0:]
                    self.coef = np.transpose(self.theta[1:,0:])

    def predict(self,x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)),x]
        z = np.dot(x_b, self.theta)
        prob = self.sigmoidea(z)
        y_pred = np.where(prob>=self.umbral,1,0)
        return y_pred