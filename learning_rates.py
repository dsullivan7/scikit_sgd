from abc import ABCMeta, abstractmethod


def get_learning_rate(learning_rate_type, eta0=.01):
    return {"static": Static,
            "exponential": Exponential
            }[learning_rate_type.lower()](eta0)


class BaseLearningRate(object):
    __metaclass__ = ABCMeta

    def __init__(self, eta0):
        self.eta0 = eta0

    @abstractmethod
    def step(self, num_iter):
        pass


class Static(BaseLearningRate):
    def step(self, num_iter):
        return self.eta0


class Exponential(BaseLearningRate):
    def step(self, num_iter):
        return (1. + .02 * num_iter) ** (-2. / 3.)
