import numpy as np
import loss_functions_slow
import learning_rates_slow
import sgd_opt

# remove after debugging
# import pdb


class NewSGD():

    def __init__(self,
                 loss,
                 eta0=.001,
                 learning_rate_type="static",
                 n_iter=5,
                 avg=False,
                 callback=None,
                 alpha=0.):
        self.loss = loss
        self.n_iter = n_iter
        self.eta0 = eta0
        self.avg = avg
        self.learning_rate = \
            learning_rates_slow.get_learning_rate(learning_rate_type, eta0)
        self.callback = callback
        self.alpha = alpha

    def fit(self, X, y):
        self._fit(X,
                  y,
                  self.coef_,
                  self.loss,
                  self.eta0,
                  self.learning_rate,
                  self.n_iter)
        return self

    def partial_fit(self, X, y):
        # set all class variables
        if not hasattr(self, "pobj_"):
            self.pobj_ = []
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(X.shape[1])
        if not hasattr(self, "coef_avg_") and self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])
        if not hasattr(self, "total_iter_"):
            self.total_iter_ = 0

        sgd_opt.partial_fit(X,
                            y,
                            self.coef_,
                            n_iter,
                            self.eta0)
        return self

    def _fit(self, X, y, loss, eta0, learning_rate, n_iter):
        # initialize components needed for weight calculation
        self.coef_ = np.zeros(X.shape[1])
        self.total_iter_ = 0
        self.pobj_ = []

        # components for asgd
        if self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])

        return sgd_opt.partial_fit(X, y, n_iter, eta0)
        # return self._partial_fit(X, y, loss, eta0, learning_rate, n_iter)

    def _partial_fit(self, X, y, loss, eta0, learning_rate, n_iter):
        # set all class variables
        if not hasattr(self, "pobj_"):
            self.pobj_ = []
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(X.shape[1])
        if not hasattr(self, "coef_avg_") and self.avg:
            self.coef_avg_ = np.zeros(X.shape[1])
        if not hasattr(self, "total_iter_"):
            self.total_iter_ = 0

        # initialize components needed for weight calculation
        weights = np.copy(self.coef_)
        loss_function = loss_functions_slow.get_loss_function(loss)
        total_iter = self.total_iter_
        pobj = []  # stores total loss for each iteration
        alpha = self.alpha

        # components for asgd
        if self.avg:
            avg_weights = self.coef_avg_

        # iterate according to the number of iterations specified
        for n in range(n_iter):

            # iterate over each entry point in the training set
            for i in range(X.shape[0]):
                total_iter += 1

                # base sgd code
                p = np.dot(X[i], weights)
                gradient = loss_function.dloss(p, y[i]) * \
                    X[i] + alpha * weights
                step = self.learning_rate.step(num_iter=total_iter,
                                               gradient=gradient)
                weights += step
                # averaged sgd
                if self.avg:
                    avg_weights *= total_iter - 1
                    avg_weights += weights
                    avg_weights /= total_iter

                # loss calculation
                if total_iter % 1 == 0:
                    if self.avg:
                        pobj.append(self.callback(avg_weights, alpha))
                    else:
                        pobj.append(self.callback(weights, alpha))

        # set the corresponding private values
        self.total_iter_ = total_iter
        self.pobj_ += pobj
        self.coef_ = weights
        if self.avg:
            self.coef_avg_ = avg_weights

        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sgd_jit

    n_iter = 1

    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 1000, 10

    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # Define a ground truth on the scaled data
    y = np.dot(X, w)
    y = np.sign(y)
    """
    """
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Make it binary
    X = X[y < 2]
    y = y[y < 2]

    y[y == 0] = -1

    """
    # """
    from sklearn import datasets

    digits = datasets.load_digits()
    X = np.array(digits.data, dtype=np.float64)
    y = np.array(digits.target, dtype=np.float64)

    # X = X[y < 2]
    # y = y[y < 2]
    y[y <= 4] = -1
    y[y > 4] = 1

    # """
    """
    nrows = 100000
    X = np.array(p.read_table('../leon_sgd/sgd/data/pascal/alpha_train.dat',
                              sep=" ",
                              nrows=nrows,
                              header=None))[:,:-1]
    y = np.array(p.read_table('../leon_sgd/sgd/data/pascal/alpha_train.lab',
                              nrows=nrows,
                              header=None))
    """
    chunks = 1
    x_chunks = np.array_split(X, chunks)
    y_chunks = np.array_split(y, chunks)

    loss = 'hinge'
    # loss = 'log'

    alpha = 0
    # alpha = 1e-5

    def callback(coef, alpha=0.):
        loss_function = loss_functions_slow.get_loss_function(loss)
        pobj = np.mean(list(map(loss_function.loss,
                            np.dot(X, coef) +
                            (alpha * np.dot(coef, coef) / 2.),
                            y)))
        return pobj

    import time
    model = NewSGD(loss,
                   learning_rate_type='static',
                   eta0=.1,
                   n_iter=n_iter,
                   avg=False,
                   alpha=alpha,
                   callback=callback)
    time1 = time.time()
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        model.partial_fit(x_chunk, y_chunk)
    time2 = time.time()
    print("the module without numba took: "
          + str(time2 - time1) + " seconds")

    time1 = time.time()
    jit_model = sgd_jit.JitSGD(loss,
                               learning_rate_type='static',
                               eta0=.1,
                               n_iter=n_iter,
                               avg=False,
                               alpha=alpha,
                               callback=callback)
    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        jit_model.partial_fit(x_chunk, y_chunk)
    time2 = time.time()
    print("the module with numba took: "
          + str(time2 - time1) + " seconds")

    """
    avg_model = NewSGD(loss,
                       eta0=0.1,
                       learning_rate_type='static',
                       n_iter=n_iter,
                       avg=True,
                       alpha=alpha,
                       callback=callback)

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        avg_model.partial_fit(x_chunk, y_chunk)

    adagrad_model = NewSGD(loss,
                           eta0=.001,
                           learning_rate_type='adagrad',
                           n_iter=n_iter,
                           avg=False,
                           alpha=alpha,
                           callback=callback)

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        adagrad_model.partial_fit(x_chunk, y_chunk)

    adadelta_model = NewSGD(loss,
                            eta0=.01,
                            learning_rate_type='adadelta',
                            n_iter=n_iter,
                            avg=False,
                            alpha=alpha,
                            callback=callback)

    for x_chunk, y_chunk in zip(x_chunks, y_chunks):
        adadelta_model.partial_fit(x_chunk, y_chunk)

    # from sag import SAG
    # sag_model = SAG(loss, alpha=alpha, n_iter=n_iter, callback=callback)
    # sag_model.fit(X, y)
    """
    """
    npinto_model = naive_asgd.NaiveBinaryASGD(n_features,
                                              sgd_step_size0=.01,
                                              n_iterations=iterations)
    """
    # npinto_model.fit(np.array(X), np.array(y))
    loss_function = loss_functions_slow.get_loss_function(loss)

    if loss == 'log':
        from sklearn.linear_model import LogisticRegression
        # w_opt, pobj_opt = newton_logistic(X, y, alpha=alpha)
        lr = LogisticRegression(fit_intercept=False,
                                tol=1e-9,
                                C=1./(alpha * X.shape[0]))
        w_opt = lr.fit(X, y).coef_.ravel()
        pred = np.dot(X, w_opt)
        pred += alpha * np.dot(w_opt, w_opt) / 2.
        pobj_opt = np.mean(map(loss_function.loss, pred, y))

    w_jit = jit_model.coef_.ravel()
    pred1 = np.dot(X, w_jit)
    jit = np.mean(list(map(loss_function.loss, pred1, y)))
    print("jit: " + str(jit))

    w_reg = model.coef_.ravel()
    pred2 = np.dot(X, w_reg)
    reg = np.mean(list(map(loss_function.loss, pred2, y)))
    print("reg: " + str(reg))
    """
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(n_iter=n_iter, alpha=alpha, eta0=.01)
    w_opt = clf.fit(X, y).coef_.ravel()
    pred = np.dot(X, w_opt)
    pred += alpha * np.dot(w_opt, w_opt) / 2.
    pobj_opt = np.mean(map(loss_function.loss, pred, y))
    """

    plt.close('all')
    # plt.plot(model.pobj_, label='SGD')
    # plt.plot(avg_model.pobj_, label='ASGD')
    # plt.plot(adagrad_model.pobj_, label='ADAGRAD')
    # plt.plot(adadelta_model.pobj_, label='ADADELTA')
    # plt.plot(sag_model.pobj_, label='SAG')
    # plt.axhline(pobj_opt, label='OPT', linestyle='--', color='k')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()

    # plt.figure()
    # print(zip(np.sign(np.dot(X, adagrad_model.coef_)), y))
    plt.plot(model.pobj_, label='SGD')
    plt.axhline(jit, label='JIT', linestyle='--', color='k')
    # plt.plot(np.log10(avg_model.pobj_), label='ASGD')
    # plt.plot(np.log10(adagrad_model.pobj_), label='ADAGRAD')
    # plt.plot(np.log10(adadelta_model.pobj_), label='ADADELTA')
    # plt.plot(np.log10(sag_model.pobj_), label='SAG')
    # plt.axhline(np.log10(pobj_opt), label='OPT', linestyle='--', color='k')
    # plt.plot(np.log10(npinto_model.pobj_), label='NPINTO')
    plt.xlabel('iter')
    plt.ylabel('cost')
    plt.legend()
    # plt.show()
