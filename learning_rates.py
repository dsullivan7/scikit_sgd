from abc import ABCMeta, abstractmethod
import numpy as np


def get_learning_rate(learning_rate_type, eta0=.01):
    return {"static": Static,
            "exponential": Exponential,
            "adagrad": AdaGrad
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
        return self.eta0


class Exponential(BaseLearningRate):
    def step(self, num_iter, gradient=None):
        return (1. + .02 * num_iter) ** (-2. / 3.)


class AdaGrad(BaseLearningRate):
    def __init__(self, eta0):
        self.eta0 = eta0
        self.sum_squared_grad = 0

    def step(self, gradient, num_iter=None):
        self.sum_squared_grad += gradient ** 2
        return self.eta0 / np.sqrt(self.sum_squared_grad)
