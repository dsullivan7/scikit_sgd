import numpy as np
import learning_rates_slow
from numba import jit

# remove after debugging
# import pdb


class JitSGD():

    def __init__(self,
                 loss,
                 eta0=.001,
                 learning_rate_type="static",
                 n_iter=5,
                 avg=False,
                 callback=None,
                 alpha=0.):
        self.loss = loss
        self.n_iter = n_iter
        self.eta0 = eta0
        self.avg = avg
        self.learning_rate = \
            learning_rates_slow.get_learning_rate(learning_rate_type, eta0)
        self.callback = callback
        self.alpha = alpha

    def fit(self, X, y):
        return self._fit(X,
                         y,
                         self.loss,
                         self.eta0,
                         self.learning_rate,
                         self.n_iter)

    def partial_fit(self, X, y):
        return self._partial_fit(X,
                                 y,
                                 self.loss,
                                 self.eta0,
                                 self.learning_rate,
                                 self.n_iter)

    def _fit(self, X, y, loss, eta0, learning_rate, n_iter):
        # initialize components needed for weight calculation
        self.coef_ = np.zeros(X.shape[1])
        self.total_iter_ = 0
        self.pobj_ = []

        # components for asgd
        if self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])

        return self._partial_fit(X, y, loss, eta0, learning_rate, n_iter)

    def _partial_fit(self, X, y, loss, eta0, learning_rate, n_iter):
        # set all class variables
        if not hasattr(self, "pobj_"):
            self.pobj_ = []
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(X.shape[1])
        if not hasattr(self, "coef_avg_") and self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])
        if not hasattr(self, "total_iter_"):
            self.total_iter_ = 0

        # initialize components needed for weight calculation
        weights = np.copy(self.coef_)
        self.coef_ = self._partial_fit_iterate(X, y, weights, n_iter, eta0)
        return self

    @jit
    def _partial_fit_iterate(X, y, weights, n_iter, learning_rate):
        # iterate according to the number of iterations specified
        for n in range(n_iter):
            # iterate over each entry point in the training set
            for i in range(X.shape[0]):
                p = np.dot(X[i], weights)
                z = p * y[i]
                step = 0
                if z <= 1.0:
                    step = -y[i]
                gradient = step * X[i]
                update = -learning_rate * gradient
                weights += update

        return weights
