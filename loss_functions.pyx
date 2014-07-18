from abc import ABCMeta, abstractmethod

def get_loss_function(loss):
    return {"hinge": Hinge,
            }[loss.lower()]()

class BaseLossFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, p, y):
        pass

    @abstractmethod
    def dloss(self, p, y):
        pass


class Hinge(BaseLossFunction):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def loss(self, p, y):
        z = p * y
        if z <= self.threshold:
            return self.threshold - z
        else:
            return 0

    def dloss(self, p, y):
        z = p * y
        if z <= self.threshold:
            return -y
        else:
            return 0
