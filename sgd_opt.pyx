import numpy as np
import loss_functions

def partial_fit(X, y, n_iter, eta0):
    cdef int i
    cdef int n
    cdef double p
    weights = np.zeros(X.shape[1])
    loss_function = loss_functions.get_loss_function("hinge")
    # iterate over a constant n
    for n in range(n_iter):
        # iterate over each sample in X
        for i in range(X.shape[0]):
            p = np.dot(X[i], weights)
            
            z = p * y[i]
            step = 0
            if z <= 1.0:
                 step = -y[i]
            gradient = step * X[i]
            #gradient = loss_function.dloss(p, y[i]) * X[i]
            weights += -eta0 * gradient

    return weights



