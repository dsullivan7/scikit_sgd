cimport loss_functions
import loss_functions
cimport numpy as np

def partial_fit(np.ndarray[double, ndim=2, mode="c"] X, np.ndarray[double, ndim=1, mode="c"]  y, np.ndarray[double, ndim=1, mode="c"]  weights, int n_iter, double eta0):
    _partial_fit(&X[0, 0], &y[0], &weights[0], n_iter, eta0, X.shape[0], X.shape[1])

cdef void _partial_fit(double* X, double* y, double* weights, int n_iter, double eta0, int n_samples, int n_features):
    cdef int total_iter = 0
    cdef int n
    cdef int i
    cdef int j
    cdef double p
    cdef double gradient

    cdef loss_functions.Hinge loss_function = loss_functions.get_loss_function()
    # iterate over a constant n which is the number of times
    # for iterating over the total number of samples
    for n in range(n_iter):

        # iterate over each sample in X
        for i in range(n_samples):
            total_iter = i * n_features

            # compute the dot product of the weights and the current sample
            p = 0
            for j in range(n_features):
                p = p + weights[j] * X[total_iter + j]

            # compute the gradient and update the weights
            for j in range(n_features):
                gradient = loss_function.dloss(p, y[i]) * X[total_iter + j]
                weights[j] = weights[j] - (eta0 * gradient)
