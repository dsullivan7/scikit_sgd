from sgd import NewSGD
from random import randint
import numpy as np

def test_something():
    clf = NewSGD(loss="SquaredLoss",eta0=.000001,n_iter=10)
    x = 5
    y = 152
    arr = np.empty([y,x])
    
    for i in range(y):
        arr[i,0] = i
        for j in range(1,x):
            arr[i][j] = randint(10,20)
    clf.fit(arr, arr[:,0])

    print(clf.coef_)
