import sgd
import loss_functions_slow
from sklearn import datasets

import numpy as np

if __name__ == "__main__":
    digits = datasets.load_digits()
    X = np.array(digits.data, dtype=np.float64)
    y = np.array(digits.target, dtype=np.float64)
    y[y <= 4] = -1
    y[y > 4] = 1

    learning_rate = .01
    n_iter = 1
    model = sgd.NewSGD("hinge", n_iter=n_iter, eta0=learning_rate)
    loss_function = loss_functions_slow.get_loss_function("hinge")

    model.partial_fit(X, y)
    w_reg = model.coef_.ravel()
    pred = np.dot(X, w_reg)
    reg = np.mean(list(map(loss_function.loss, pred, y)))

    print("danny implementation: " + str(reg))
