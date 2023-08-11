import numpy as np
from sklearn.base import BaseEstimator

class LogisticRegression(BaseEstimator):

    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step 

    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    def predict(self, X):
        res = self.__predict(X)
        res[res >= 0] = 1
        res[res < 0] = 0

        return res

    def der_reg(self):
        return -self.w
    
    def der_loss(self, x, y):
        features = 20
        x.shape == (self.batch_size, features)
        y.shape == (self.batch_size,)

        der_w = np.zeros((features))
        der_w0 = 0.

        for i in range(self.batch_size):
            sig = np.exp(-y[i] * (self.__predict(x[i])))
            M = sig / (1. + sig)
            der_w += -x[i] * y[i] * M
            der_w0 += -y[i] * M

        return der_w / self.batch_size, der_w0 / self.batch_size

    def fit(self, X_train, y_train):

        random_gen = np.random.RandomState(self.random_state)
        size, dim = X_train.shape

        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):

            rand_indices = random_gen.choice(size, self.batch_size)

            x = X_train[rand_indices]
            y = y_train[rand_indices]

            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg() * self.C 

            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step 

        return self
