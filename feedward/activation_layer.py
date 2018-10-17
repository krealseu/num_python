import numpy as np

from .layer import ILayer

__all__ = ['activefunc']


def sigmoid(x): return 1. / (1 + np.exp(-x))


def sigmoid_back(y, diff): return diff * y * (1 - y)


def tanh(x): return np.tanh(x)


def tanh_back(y, diff): return diff * (1 - y ** 2)


def relu(x): return np.maximum(0, x)


def relu_back(y, diff): return np.where(y > 0, diff, 0)


def prelu(x, r=0.01): return np.maximum(x * r, x)


def prelu_back(y, diff, r=0.01): return np.where(y > 0, diff, diff * r)


def trelu(x, r=0.01): return np.minimum(np.maximum(x * r, x),x*r+(1-r))

def trelu_back(y, diff, r=0.01): 
    tmp = np.where(y > 0, diff, diff * r)
    tmp = np.where(y < 1, tmp, tmp * r)
    return tmp


def elu_relu(x, r=0.01): return np.where(x > 0, x, (np.exp(x) - 1) * r)


def elu_relu_back(y, diff, r=0.01): return np.where(y > 0, diff, diff * (1 - y) * r)


funcs = {'relu': [relu, relu_back],
         'prelu': [prelu, prelu_back],
         'trelu': [trelu, trelu_back],
         'elu': [elu_relu, elu_relu_back],
         'tanh': [tanh, tanh_back],
         'sigmoid': [sigmoid, sigmoid_back]}


class ActivationLayer(ILayer):
    def __init__(self, name, forward, backward, *params):
        super().__init__()
        self.__name = name
        self.__params = params
        self.__forwardFunc = forward
        self.__backwardFunc = backward
        return

    def predict(self, x): return self.__forwardFunc(x, *self.__params)

    def forward(self, x):
        self.y = self.__forwardFunc(x, *self.__params)
        return self.y

    def backward(self, diff):
        assert diff.shape == self.y.shape
        return self.__backwardFunc(self.y, diff, *self.__params)

    pass


def activefunc(name: str, *param):
    """
    提供相应名称的激活函数层
    :param name: 激活函数的名称 sigmoid , tanh , relu , prelu , elu
    :parameter param: 激活函数的参数，sigmoid tanh relu 无参数，prelu elu 有一个参数 默认值0.01
    :return: 返回相应激活功能的处理层
    """
    try:
        func = funcs[name]
        return ActivationLayer(name, func[0], func[1], *param)
    except KeyError:
        raise NameError("error name")
