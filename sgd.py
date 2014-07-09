import numpy as np
import pdb
from scipy import linalg


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

    def _fit(self, X, y, loss, learning_rate, n_iter):
        self._fit_regressor(X, y, loss, learning_rate, n_iter)
        return self

    def _get_loss_function(self, loss):
        return {"SquaredLoss": SquaredLoss,
                "Hinge": Hinge
                }[loss]()

    def _fit_regressor(self, X, y, loss, learning_rate, n_iter):
        # initialize components needed for weight calculation
        weights = np.zeros(X.shape[1])
        loss_function = self._get_loss_function(loss)
        learning_rate_type = learning_rate

        # components for asgd
        avg_weights = np.zeros(X.shape[1])
        total_iter = 0
        pobj = []

        # iterate according to the number of iterations specified
        for n in range(n_iter):
            for i in range(X.shape[0]):

                # base sgd code
                p = np.dot(X[i], weights)
                update = loss_function.dloss(p, y[i])
                weights -= learning_rate_type * update * X[i]

                # asgd
                if self.avg:
                    # pdb.set_trace()
                    avg_weights *= total_iter
                    avg_weights += weights
                    avg_weights /= (total_iter + 1)
                    weights = np.copy(avg_weights)
                    total_iter += 1

                loss = 0.5 * linalg.norm(y - np.dot(X, weights)) ** 2
                pobj.append(loss)

        # set the corresponding private values
        self.pobj_ = pobj
        self.coef_ = weights


class Hinge():
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def loss(self, p, y):
        z = p * y
        return self.threshold - z if z <= threshold else return 0

    def dloss(self, p, y):
        z = p * y
        return -y if z <= threshold else return 0


class SquaredLoss():
    def loss(self, p, y):
        return .5 * (p - y) * (p - y)

    def dloss(self, p, y):
        return p - y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import naive_asgd
    import experimental_asgd

    rng = np.random.RandomState(42)
    n_samples, n_features = 1000, 10

    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))

    # Define a ground truth on the scaled data
    y = np.dot(X, w)

    iterations = 5

    model = NewSGD('Hinge', eta0=.01, n_iter=iterations, avg=False)
    model.fit(X, y)

    avg_model = NewSGD('Hinge', eta0=1., n_iter=iterations, avg=True)
    avg_model.fit(X, y)

    npinto_model = naive_asgd.NaiveBinaryASGD(n_features,
                                              sgd_step_size0=.01,
                                              n_iterations=iterations)
    npinto_model.fit(X, y)

    npinto_model2 = experimental_asgd.ExperimentalBinaryASGD(n_features)
    npinto_model2.fit(X, y)

    plt.close('all')
    plt.plot(np.log10(model.pobj_), label='SGD')
    plt.plot(np.log10(avg_model.pobj_), label='ASGD')
    plt.plot(np.log10(npinto_model.pobj_), label='NPINTO')
    plt.plot(np.log10(npinto_model2.pobj_), label='NPINTO2')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
