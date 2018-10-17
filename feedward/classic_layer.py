import numpy as np

from .layer import ILayer

__all__ = ['SVM', 'Softmax', 'Logistic']


class ClassicLayer(ILayer):
    def __init__(self): super().__init__()

    def loss(self, ty): return

    pass


class Softmax(ClassicLayer):

    def __init__(self, ):
        super().__init__()
        self.shift = self.sum = 0
        return

    def predict(self, x: np.ndarray):
        exp = np.exp(x - x.max(axis=1, keepdims=True))
        sum_exp = exp.sum(axis=1, keepdims=True)
        return (exp / sum_exp > 0.5) * 1

    def forward(self, x: np.ndarray):
        shift = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shift)
        sum_exp = exp.sum(axis=1, keepdims=True)
        self.x, self.shift, self.sum, self.y = x, shift, sum_exp, exp / sum_exp
        return self.y

    def backward(self, ty: np.ndarray, back_diff=True):
        assert ty.shape == self.y.shape
        return (self.y - ty) / ty.shape[0]

    def loss(self, ty):
        return ((np.log(self.sum) - self.shift) * ty).sum(axis=-1).mean()

    pass


class SVM(ClassicLayer):
    def predict(self, x: np.ndarray): return (x == x.max(axis=1, keepdims=True)) * 1

    def forward(self, x):
        self.x, self.y = x, x
        return self.y

    def backward(self, ty, back_diff=True):
        assert ty.shape == self.y.shape
        tmp = (np.maximum(0, self.y - (self.y * ty).sum(axis=1, keepdims=True) + 1))
        return (tmp - tmp.sum(axis=1, keepdims=True) * ty) / ty.shape[0]

    def loss(self, ty):
        return np.maximum(0, self.y - (self.y * ty).sum(axis=1, keepdims=True) + 1).sum(axis=-1).mean() - 1

    pass


class Logistic(ClassicLayer):

    def predict(self, x: np.ndarray):
        return (1. / (1 + np.exp(-x)) > 0.5) * 1

    def forward(self, x):
        self.x, self.y = x, 1. / (1 + np.exp(-x))
        return self.y

    def backward(self, ty, back_diff=True):
        assert ty.shape == self.y.shape
        return (self.y - ty) / ty.shape[0]

    def loss(self, ty): return ((self.y - ty) ** 2).mean()

    pass
