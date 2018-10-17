import numpy as np

from .activation_layer import *
from .classic_layer import *
from .connection_layer import *
from .optimizes import GSD

__all__ = ['Model']


class Model:
    def __init__(self):
        self.layers = []
        self.eta = 1
        pass

    def add(self, i_layer):
        self.layers.append(i_layer)
        pass

    def train(self, x, y, epochs: int = 10, batch_size: int = 0):
        assert x.shape[0] == y.shape[0]
        train_num = x.shape[0]
        if batch_size == 0:
            batch_size = train_num
        one_circle = range(0, int(train_num / batch_size))
        for i in range(0, epochs):
            aa = []
            for k in one_circle:
                start = k * batch_size % train_num
                train_x = x[start:start + batch_size, :]
                target = y[start:start + batch_size, :]
                _ = self.__feedforward(train_x)
                loss = self.layers[-1].loss(target)
                aa.append(loss)
                self.__backprop(target)
                self.__update()
                pass
            info = "epochs={}/{} loss={}".format(i, epochs, np.mean(aa))
            print(info)

    def __feedforward(self, input_data):
        x = input_data
        for l in self.layers:
            x = l.forward(x)
        return x

    def __backprop(self, input_diff):
        diff = input_diff
        for l in reversed(self.layers[1:]):
            diff = l.backward(diff)
        self.layers[0].backward(diff, False)
        return

    def __update(self):
        for l in self.layers:
            l.update(self.eta)
        return

    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x


class Model2:
    def __init__(self, layers: list):
        self.layers = layers
        self.optimize = GSD()
        pass

    def __train_one(self, x, y):
        py = x
        for l in self.layers:
            py = l.forward(py)
            pass
        loss = self.layers[-1].loss(y)
        # for l in reversed(self.layers[1:]):
        #     diff = l.backward(diff)
        #
        # self.layers[0].diff = diff
        return loss

    def train(self, x, y, epochs: int = 10, batch_size: int = 0):
        assert x.shape[0] == y.shape[0]
        train_num = x.shape[0]
        if batch_size == 0:
            batch_size = train_num
        one_circle = range(0, int(train_num / batch_size))
        for i in range(0, epochs):
            aa = []
            for k in one_circle:
                start = k * batch_size % train_num
                train_x = x[start:start + batch_size, :]
                target = y[start:start + batch_size, :]
                _ = self.__feedforward(train_x)
                loss = self.layers[-1].loss(target)
                aa.append(loss)
                self.__backprop(target)
                self.__update()
                pass
            info = "epochs={}/{} loss={}".format(i, epochs, np.mean(aa))
            print(info)

    def __feedforward(self, input_data):
        x = input_data
        for l in self.layers:
            x = l.forward(x)
        return x

    def __backprop(self, input_diff):
        diff = input_diff
        for l in reversed(self.layers[1:]):
            diff = l.backward(diff)
        self.layers[0].diff = diff
        return

    def __update(self):
        for l in self.layers:
            l.update(self.eta)
        return

    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x


class NeuralNetwork(Model):

    def __init__(self, *layers_num):
        super(NeuralNetwork, self).__init__()
        for i in range(len(layers_num)):
            if i == 0:
                continue
            else:
                ii, o = layers_num[i - 1], layers_num[i]
                if i == len(layers_num) - 1:
                    self.add(FullConnection(ii, o))
                    if o == 1:
                        self.add(Logistic())
                    else:
                        self.add(Softmax())
                else:
                    self.add(FullConnection(ii, o))
                    self.add(Full(o))
#                     self.add(activefunc('prelu'))
                    pass
        pass
