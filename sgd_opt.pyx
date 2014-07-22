cimport loss_functions
import numpy as np
cimport numpy as np

# import cblas module
cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double*, int, double*, int) nogil

# the entry point from python into the cython
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
    loss_function.set_threshold(1.0)

    # iterate over a constant n which is the number of times
    # for iterating over the total number of samples
    for n in range(n_iter):

        # iterate over each sample in X
        for i in range(n_samples):
            total_iter = i * n_features

            # compute the dot product of the weights and the current sample
            p = ddot(n_features, weights, 1, X + total_iter, 1)
            
            # compute the gradient and update the weights
            for j in range(n_features):
                gradient = loss_function.dloss(p, y[i]) * X[total_iter + j]
                weights[j] = weights[j] - (eta0 * gradient)
