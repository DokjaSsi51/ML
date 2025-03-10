import numpy as np

class Linear():
    def __init__(self):
        self.theta = None
        self.intercept = None
        self.coef = None
        
    def fit(self, x, y): 
        n = len(x)
        x_b = np.c_[np.ones((n, 1)), x] #Se concatenan para darle un valor más a x para que coincidieran las dimensiones
        self.theta = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
        self.intercept = self.theta[0]
        self.coef = self.theta[1:]

    def predict(self, x):
        n = len(x)
        x_b = np.c_[np.ones((n, 1)), x] #Concatena una columna de unos con los datos que se ingresen
        y_pred = x_b.dot(self.theta)
        return y_pred

class DGRegression():
    def __init__(self, epochs = 1000, mini_lote = None, semilla = None, eta0 = 0.1, pacience = 10, tol = 1e-3): 
        self.mini_lote = mini_lote
        self.semilla = semilla
        self.eta0 = eta0 #El tamaño del paso
        self.epochs = epochs #numero de iteraciones
        self.theta = None
        self.intercept = None
        self.coef = None
        self.pacience = pacience
        self.tol = tol

    def aprendizaje(self, epoch, eta0):
        return (eta0/(1+epoch))
    
    def predict(self, x):
        n = len(x)
        x_b = np.c_[np.ones((n, 1)), x] #Concatena una columna de unos con los datos que se ingresen
        y_pred = x_b.dot(self.theta)
        return y_pred

    def calcular_error(self, x_b, y):
        prediction = x_b.dot(self.theta)
        error = np.mean((prediction - y)**2)
        return error

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        self.theta = np.random.rand(n+1, 1)
        best_error = float("inf")
        pacience_counter = 0
        
        for epoch in range(self.epochs):
            if (self.mini_lote is None and self.semilla is None ):
                #eta = self.aprendizaje(i, self.eta0) #Tasa de aprendizaje
                grad = (2/m)*(x_b.T.dot(x_b.dot(self.theta) - y))
                    
            elif (self.mini_lote is not None and self.semilla is None):
                index_m = np.random.permutation(m)
                x_bm = x_b[index_m]
                y_m = y[index_m]

                for j in range(0, m, self.mini_lote):
                    xi = x_bm[j:j + self.mini_lote]
                    yi = y_m[j:j + self.mini_lote]

                    eta = self.eta0
                    grad = (2/self.mini_lote)*(xi.T.dot(xi.dot(self.theta)-yi))
    
            elif (self.mini_lote is None and self.semilla is not None):
                for j in range(m):
                    random_index = np.random.randint(m)
                    xi = x_b[random_index:random_index + 1]
                    yi = y[random_index:random_index + 1]
                    
                    grad = 2*(xi.T.dot(xi.dot(self.theta) - yi))
                    
            self.theta = self.theta - self.aprendizaje(epoch, self.eta0)*grad
            self.intercept = self.theta[0]
            self.coef = self.theta[1:]
            
            currenly_error = self.calcular_error(x_b, y)
            
            if (abs(best_error - currenly_error) < self.tol): 
                print(f"Se detuvo por convergencia en iteración {epoch + 1}")
                break
                
            if (currenly_error  < best_error): 
                best_error = currenly_error
                pacience_counter = 0
            else:
                pacience_counter += 1

            if (pacience_counter >= self.pacience):
                print(f"Se detuvo por convergencia por detención anticipada en la iteración {epoch + 1}")
                break

    def predict(self, x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        y_pred = x_b.dot(self.theta)
        return y_pred

class Ridge:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.theta = None
        self.intercept = None
        self.coef = None

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        I = np.identity(n+1)
        self.theta = np.linalg.inv(x_b.T.dot(x_b)+self.alpha*I).dot(x_b.T).dot(y)
        self.intercept = self.theta[0]
        self.coef = self.theta[1:]

    def predict(self,x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        y_pred = x_b.dot(self.theta)
        return y_pred

class Lasso:
    def __init__(self, alpha = 0.1, epochs = 1000, tol = 1e-3):
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        self.theta = None
        self.intercept = None
        self.coef = None 

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        self.theta = np.zeros((n+1,1))
        #for epoch in range(self.epochs):
        theta_p = self.theta.copy()
        for i in range(n):
            if i == 0:
                gra = 2*x_b[:,0].dot(x_b.dot(self.theta)-y)
            else:
                gra = 2*x_b[:,i].dot(x_b.dot(self.theta)-y) + self.alpha*np.sign(self.theta[i])
            self.theta[i] = self.theta[i]-self.alpha*gra
        #if (np.linalg.norm(self.theta - theta_p) < self.tol):
            #break

        self.intercept = self.theta[0]
        self.coef = self.theta[1:]

    def predict(self,x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        y_pred = x_b.dot(self.theta)
        return y_pred