import numpy as np


class GSD:
    def __init__(self, beta1=0.9, beta2=0.995, lr=1, r2=1e-5):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.regularization = r2
        self.m = {}
        self.v = {}
        return

    def register(self, key):
        self.m[key] = 0
        self.v[key] = 0

    def update(self, key, value, gradient):
        dw = gradient + value * self.regularization
        self.m[key] = self.m[key] * self.beta1 + dw * (1 - self.beta1)
        self.v[key] = self.v[key] * self.beta2 + dw ** 2 * (1 - self.beta2)
        return value - self.lr * self.m[key] / (np.sqrt(self.v[key]) + self.lr)
