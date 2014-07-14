import numpy as np
import math
import loss_functions
from scipy import linalg

# remove after debugging
import pdb


class NewSGD():

    def __init__(self, loss, eta0=.001, n_iter=5, avg=False):
        self.loss = loss
        self.n_iter = n_iter
        self.eta0 = eta0
        self.avg = avg

    def fit(self, X, y):
        return self._fit(X,
                         y,
                         self.loss,
                         self.eta0,
                         self.n_iter)

    def partial_fit(self, X, y):
        return self._partial_fit(X,
                                 y,
                                 self.loss,
                                 self.eta0,
                                 self.n_iter)

    def _fit(self, X, y, loss, learning_rate, n_iter):
        # initialize components needed for weight calculation
        self.coef_ = np.zeros(X.shape[1])
        self.total_iter_ = 0
        self.pobj_ = []

        # components for asgd
        if self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])
        
        return self._partial_fit(X, y, loss, learning_rate, n_iter)

    def _partial_fit(self, X, y, loss, learning_rate, n_iter):
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
        loss_function = loss_functions.get_loss_function(loss)
        total_iter = self.total_iter_
        pobj = [] # stores total loss for each iteration

        # components for asgd
        if self.avg:
            avg_weights = self.coef_avg_

        # iterate according to the number of iterations specified
        for n in range(n_iter):
            # iterate over each entry point in the training set
            for i in range(X.shape[0]):
                total_iter += 1

                # base sgd code
                p = np.dot(X[i], weights)
                update = loss_function.dloss(p, y[i])
                weights -= learning_rate * update * X[i]

                # averaged sgd
                if self.avg:
                    avg_weights *= total_iter - 1
                    avg_weights += weights
                    avg_weights /= total_iter

                # loss calculation
                if self.avg:
                    p = np.dot(X, avg_weights)
                else:
                    p = np.dot(X, weights)
                pobj.append(sum(map(loss_function.loss, p, y)))

        # set the corresponding private values
        self.total_iter_ = total_iter
        self.pobj_ += pobj
        self.coef_ = weights
        if self.avg:
            self.coef_avg_ = avg_weights

        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as p
    import naive_asgd

    iterations = 1

    # """
    rng = np.random.RandomState(42)
    n_samples, n_features = 500, 5

    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))

    # Define a ground truth on the scaled data
    y = np.dot(X, w)
    y = np.sign(y)
    # """

    """
    nrows = 100000
    X = np.array(p.read_table('../leon_sgd/sgd/data/pascal/alpha_train.dat',
                              sep=" ",
                              nrows=nrows,
                              header=None))[:,:-1]
    y = np.array(p.read_table('../leon_sgd/sgd/data/pascal/alpha_train.lab',
                              nrows=nrows,
                              header=None))
    """

    model = NewSGD('hinge', eta0=.01, n_iter=iterations, avg=False)
    model.fit(X, y)

    avg_model = NewSGD('hinge', eta0=.1, n_iter=iterations, avg=True)
    avg_model.fit(X, y)

    """
    npinto_model = naive_asgd.NaiveBinaryASGD(n_features,
                                              sgd_step_size0=.01,
                                              n_iterations=iterations)
    """
    # npinto_model.fit(np.array(X), np.array(y))

    plt.close('all')
    plt.plot(np.log10(model.pobj_), label='SGD')
    plt.plot(np.log10(avg_model.pobj_), label='ASGD')
    # plt.plot(np.log10(npinto_model.pobj_), label='NPINTO')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
