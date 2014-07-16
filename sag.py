import math
import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

import loss_functions


def newton_logistic(X, y, alpha=1.):
    X = np.asarray(X, dtype=np.float)
    y = np.asarray(y, dtype=np.float)

    n_samples, n_features = X.shape

    R =  np.sqrt(np.max(np.sum(X ** 2, axis=1)))

    # global minimum with Newton method
    w_global = np.zeros(n_features)
    pobj_global = []

    for k in xrange(30):
        ywTx = y * np.dot(X, w_global)
        temp = 1. / (1. + np.exp(ywTx))
        grad = -1. / n_samples * np.dot(X.T, (y * temp)) + (1e-12 + alpha) * w_global
        hess = 1. / n_samples * np.dot(X.T, (temp * ( 1. - temp ))[:, None] * X)
        hess.flat[::n_features + 1] += alpha + 1e-12 * R * R
        w_global -= linalg.solve(hess, grad)

        pobj_global_i = np.mean(np.log( 1. + np.exp( - y * np.dot(X, w_global))))
        pobj_global_i += alpha * np.dot(w_global, w_global) / 2.
        pobj_global.append(pobj_global_i)

    print "Global minimum : %s" % pobj_global[-1]

    return w_global, pobj_global[-1]


class SAG(BaseEstimator):

    def __init__(self, loss, step_size=.001, n_iter=5, alpha=1., random_state=None,
                 callback=None):
        self.loss = loss
        self.n_iter = n_iter
        self.step_size = step_size
        self.alpha = alpha
        self.random_state = random_state
        self.callback = callback

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float)
        y = np.asarray(y, dtype=np.float)
        alpha = self.alpha
        n_samples, n_features = X.shape
        step_size = self.step_size

        rng = check_random_state(self.random_state)
        loss_function = loss_functions.get_loss_function(self.loss)

        R =  np.sqrt(np.max(np.sum(X ** 2, axis=1)))

        # SAG
        w = np.zeros(n_features)
        grad = np.zeros(n_features)
        gradient_memory = np.zeros((n_samples, n_features))
        pobj = []
        scaling = 1. / (R ** 2) / 4. * step_size

        # iterate according to the number of iterations specified
        for i in range(self.n_iter * n_samples):
            j = int(math.floor(rng.rand(1) * n_samples))

            # compute_gradient
            new = X[j] * loss_function.dloss(np.dot(X[j], w), y[j])
            new += alpha * w

            grad += new - gradient_memory[j]
            gradient_memory[j] = new

            w -= (scaling / min(i + 1, n_samples)) *  grad

            if (i % 1000) == 0:
                if self.callback:
                    pobj.append(self.callback(w, alpha))
                else:
                    pobj_i = np.mean(map(loss_function.loss, np.dot(X, w), y))
                    pobj_i += alpha * np.dot(w, w) / 2.
                    pobj.append(pobj_i)

        # set the corresponding private values
        self.pobj_ = pobj
        self.coef_ = w

        print "SAG minimum : %s" % pobj[-1]

        return self


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.linear_model import ridge_regression, LogisticRegression

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Make it binary
    X = X[y < 2]
    y = y[y < 2]

    y[y == 0] = -1

    # alpha = .1
    alpha = 1.

    loss, step_size = 'squaredloss', 1.
    # loss, step_size = 'log', 4.
    sag = SAG(loss=loss, step_size=step_size, n_iter=30, alpha=alpha, random_state=42)
    sag.fit(X, y)

    if loss == 'squaredloss':
        loss_function = loss_functions.get_loss_function(loss)
        w_opt = ridge_regression(X, y, alpha=alpha * X.shape[0])
        pobj_opt = np.mean(map(loss_function.loss, np.dot(X, w_opt), y))
        pobj_opt += alpha * np.dot(w_opt, w_opt) / 2.
    elif loss == 'log':
        # w_opt, pobj_opt = newton_logistic(X, y, alpha=alpha)
        loss_function = loss_functions.get_loss_function(loss)
        lr = LogisticRegression(fit_intercept=False, tol=1e-9, C=1./(alpha * X.shape[0]))
        w_opt = lr.fit(X, y).coef_.ravel()
        pobj_opt = np.mean(map(loss_function.loss, np.dot(X, w_opt), y))
        pobj_opt += alpha * np.dot(w_opt, w_opt) / 2.

    print(sag.pobj_[-1] - pobj_opt)

    import matplotlib.pyplot as plt
    plt.close('all')
    plt.plot(np.log(sag.pobj_ - pobj_opt), 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Primal')
    plt.show()
