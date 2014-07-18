from abc import ABCMeta, abstractmethod
import numpy as np


def get_learning_rate(learning_rate_type, eta0=.01):
    return {"static": Static,
            "exponential": Exponential,
            "adagrad": AdaGrad,
            "adadelta": AdaDelta
            }[learning_rate_type.lower()](eta0)


class BaseLearningRate(object):
    __metaclass__ = ABCMeta

    def __init__(self, eta0):
        self.eta0 = eta0

    @abstractmethod
    def step(self, num_iter=None, gradient=None):
        pass


class Static(BaseLearningRate):
    def step(self, num_iter=None, gradient=None):
        return - self.eta0 * gradient


class Exponential(BaseLearningRate):
    def step(self, num_iter, gradient=None):
        return - ((1. + .02 * num_iter) ** (-2. / 3.)) * gradient


class AdaGrad(BaseLearningRate):
    def __init__(self, eta0, eps0=1.E-7):
        self.eta0 = eta0
        self.sum_squared_grad = 0
        self.eps0 = eps0

    def step(self, gradient, num_iter=None):
        self.sum_squared_grad += gradient ** 2 + self.eps0
        return - (self.eta0 / np.sqrt(self.sum_squared_grad)) * gradient


class AdaDelta(BaseLearningRate):
    def __init__(self, eta0, rho0=0.8, eps0=1.E-7):
        self.sum_squared_grad = 0
        self.rho0 = rho0
        self.eps0 = eps0
        self.accugrad = 0
        self.accudelta = 0

    def step(self, gradient, num_iter=None):
        agrad = self.rho0 * self.accugrad + \
            (1. - self.rho0) * gradient * gradient
        dx = - np.sqrt((self.accudelta + self.eps0) /
                       (agrad + self.eps0)) * gradient
        self.accudelta = self.rho0 * self.accudelta +\
            (1. - self.rho0) * dx * dx
        self.accugrad = agrad
        return dx
