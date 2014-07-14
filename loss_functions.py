def get_loss_function(loss):
    return {"SquaredLoss": SquaredLoss,
            "Hinge": Hinge,
            "Log": Log
            }[loss]()


class Hinge():
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


class SquaredLoss():
    def loss(self, p, y):
        return .5 * (p - y) ** 2

    def dloss(self, p, y):
        return p - y


class Log():
    def loss(self, p, y):
        z = p * y
        return math.log(1 + math.exp(-z))

    def dloss(self, p, y):
        z = p * y
        return -y / (math.exp(z) + 1.0)
