cimport loss_functions
from libc.stdlib cimport malloc
import numpy as np
cimport numpy as np

# import cblas module
cdef extern from "cblas.h":
    
    # dot product of two n elemenet vectors x and y
    double ddot "cblas_ddot"(int n, double* x, int incrx, double* y, int incry) nogil
    
    # scale the passed in n element vector x at scale
    void dscal "cblas_dscal"(int n, double scale, double* x, int incrx) nogil
    
    # copies an n element vector x to another vector y
    void dcopy "cblas_dcopy"(int n, double* x, int incrx, double* y, int incry) nogil
    
    # adds a vector x * scaler scale to another vector y
    void daxpy "cblas_daxpy"(int n, double scale, double* x, int incrx, double* y, int incry) nogil 

# the entry point from python into the cython
def partial_fit(np.ndarray[double, ndim=2, mode="c"] X,
                np.ndarray[double, ndim=1, mode="c"] y,
                np.ndarray[double, ndim=1, mode="c"] weights,
                int n_iter,
                double eta0):
    _partial_fit(&X[0, 0], &y[0], &weights[0], n_iter, eta0, X.shape[0], X.shape[1])

cdef void _partial_fit(double* X, double* y, double* weights, int n_iter, double eta0, int n_samples, int n_features):
    cdef int total_iter = 0
    cdef int n
    cdef int i
    cdef int j
    cdef double p
    cdef double gradient
    cdef int incr = 1
    cdef double* x_entry = <double*> malloc(n_features * sizeof(double))
    
    cdef loss_functions.Hinge loss_function = loss_functions.get_loss_function()
    loss_function.set_threshold(1.0)

    # iterate over a constant n which is the number of times
    # for iterating over the total number of samples
    for n in range(n_iter):

        # iterate over each sample in X
        for i in range(n_samples):
            total_iter = i * n_features
            
            # copy over the entry of x
            dcopy(n_features, X + total_iter, incr, x_entry, incr) 

            # compute the dot product of the weights and the current sample
            p = ddot(n_features, weights, incr, x_entry, incr)
           
            # scale the x_entry vector to the result of the dloss function
            dscal(n_features, eta0 * loss_function.dloss(p, y[i]), x_entry, incr)

            # update the weight vector
            daxpy(n_features, -1, x_entry, incr, weights, incr)
