from sklearn.utils import check_arrays
import numpy as np

class NewSGD():
    
    def __init__(self, loss, eta0=.001, n_iter=5):
        self.loss=loss
        self.n_iter = n_iter
        self.eta0 = eta0

    def fit(self, X, y):
        return self._fit(X, 
                         y, 
                         self.loss,
                         self.eta0, 
                         self.n_iter)

    def _fit(self, X, y, loss, learning_rate, n_iter):
        self._fit_regressor(X, y, loss, learning_rate, n_iter)
        return self
    
    def _get_loss_function(self, loss):
        return SquaredLoss() 
    
    def _fit_regressor(self, X, y, loss, learning_rate, n_iter):
        #initialize components needed for weight calculation
        weights = np.zeros(X.shape[1])
        loss_function = self._get_loss_function(loss)
        learning_rate_type = .0001 
        
        #iterate according to the number of iterations specified
        for n in range(n_iter):
            for i in range(X.shape[0]):
                p = np.dot(X[i],weights)
                update = loss_function.dloss(p,y[i])
                for j in range(weights.shape[0]):
                    weights[j] = weights[j] - (learning_rate_type * update * X[i][j])
        
        #print(weights)

        #set the corresponding private values
        self.coef_ = weights
   
class SquaredLoss():
    def loss(self, p, y):
        return .5 * (p - y) * (p - y)

    def dloss(self, p, y):
        return p - y
