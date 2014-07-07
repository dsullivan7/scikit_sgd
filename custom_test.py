from sgd import NewSGD
from random import randint
import numpy as np
import unittest

class testClass():
    x = 5
    y =1000000 

    def setUp(self):
        self.arr = np.empty([self.y,self.x])
        for i in range(self.y):
            self.arr[i,0] = i + 1
            self.arr[i,2] = i + 1
            for j in range(1,self.x):
                #arr[i][j] = i + 1
                if(j != 2):
                    self.arr[i][j] = randint(1,10)

    def test_sgd(self):
        clf = NewSGD(loss="SquaredLoss",eta0=.000000000001,n_iter=3)
        clf.fit(self.arr, self.arr[:,0])
   
        print("sgd: " + str(clf.coef_))

    def test_asgd(self):
        clf = NewSGD(loss="SquaredLoss", eta0=.000000000001,n_iter=3, avg=True)
        clf.fit(self.arr, self.arr[:,0])

        print("asgd: " + str(clf.coef_))
    
