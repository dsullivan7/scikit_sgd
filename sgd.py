import numpy as np
import loss_functions_slow
import learning_rates_slow
import sgd_opt

# remove after debugging
# import pdb


class NewSGD():

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
        self._fit(X,
                  y,
                  self.loss,
                  self.eta0,
                  self.learning_rate,
                  self.n_iter)
        return self

    def partial_fit(self, X, y):
        # set all class variables
        if not hasattr(self, "pobj_"):
            self.pobj_ = []
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(X.shape[1])
        if not hasattr(self, "coef_avg_") and self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])
        if not hasattr(self, "total_iter_"):
            self.total_iter_ = 0

        sgd_opt.partial_fit(X,
                            y,
                            self.coef_,
                            self.n_iter,
                            self.eta0)
        return self

    def _fit(self, X, y, loss, eta0, learning_rate, n_iter):
        # initialize components needed for weight calculation
        self.coef_ = np.zeros(X.shape[1])
        self.total_iter_ = 0
        self.pobj_ = []

        # components for asgd
        if self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])

        sgd_opt.partial_fit(X, y, self.coef_, n_iter, eta0)

        return self

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
        loss_function = loss_functions_slow.get_loss_function(loss)
        total_iter = self.total_iter_
        pobj = []  # stores total loss for each iteration
        alpha = self.alpha

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
                gradient = loss_function.dloss(p, y[i]) * \
                    X[i] + alpha * weights
                step = self.learning_rate.step(num_iter=total_iter,
                                               gradient=gradient)
                weights += step
                # averaged sgd
                if self.avg:
                    avg_weights *= total_iter - 1
                    avg_weights += weights
                    avg_weights /= total_iter

                # loss calculation
                if total_iter % 1 == 0:
                    if self.avg:
                        pobj.append(self.callback(avg_weights, alpha))
                    else:
                        pobj.append(self.callback(weights, alpha))

        # set the corresponding private values
        self.total_iter_ = total_iter
        self.pobj_ += pobj
        self.coef_ = weights
        if self.avg:
            self.coef_avg_ = avg_weights

        return self
