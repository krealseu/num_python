import time

import numpy as np

import feedward.NeuralNetwork as nn
import mnist_help as mh


class Timer:
    @staticmethod
    def __now():
        return time.time()

    def __init__(self):
        self.start_time = self.last_lap = self.__now()
        pass

    def lap(self, out=True):
        now = self.__now()
        rs = now - self.last_lap
        self.last_lap = now
        if out:
            print(rs)
            pass
        return rs

    def time(self):
        print(self.__now() - self.start_time)

    def reset(self):
        self.start_time = self.last_lap = self.__now()
        pass

    pass


def test(x, y, model):
    py = model.predict(x)
    delta = y - py
    num = y.shape[0]
    r2e = (delta == 1).sum() / num
    e2r = (delta == -1).sum() / num
    print(r2e, e2r)
    pass


images, labels, test_images, test_labels = mh.load_mnist()
label = np.zeros([60000, 10])
for i in range(0, 60000):
    label[i][labels[i]] = 1
    pass

test_label = np.zeros([test_labels.shape[0], 10])
for i in range(test_labels.shape[0]):
    test_label[i][test_labels[i]] = 1
    pass

c2 = nn.Model()
c2.add(nn.Conv2D((3, 1, 3, 3)))
c2.add(nn.activefunc('prelu'))
c2.add(nn.Conv2D((5, 3, 3, 3)))
c2.add(nn.activefunc('prelu'))
c2.add(nn.MaxPooling())
c2.add(nn.Flatten())
c2.add(nn.FullConnection(784 * 5, 100))
# c2.add(nn.Dropout(0.8))
c2.add(nn.activefunc('prelu'))
c2.add(nn.FullConnection(100, 50))
c2.add(nn.activefunc('prelu'))
# c2.add(nn.Relu())
c2.add(nn.FullConnection(50, 10))
c2.add(nn.SVM())

c3 = nn.Model()
c3.add(nn.FullConnection(784, 100))
c3.add(nn.Full(100))
# c3.add(nn.activefunc('prelu'))
c3.add(nn.FullConnection(100, 50))
c3.add(nn.Full(50))
# c3.add(nn.activefunc('prelu'))
c3.add(nn.FullConnection(50, 10))
c3.add(nn.SVM())

c4 = nn.NeuralNetwork(784,500,400,300,200,100,10)
c5 = nn.NeuralNetwork(784,500,500,500,400,400,300,200,100,10)
t1 = Timer()
c = c3

c.train(images, label, epochs=20, batch_size=200)
# c.train(images.reshape(-1, 1, 28, 28), label, epochs=4, batch_size=100)

t1.lap()

test(test_images, test_label, c)
# test(test_images.reshape(-1, 1, 28, 28), test_label, c)
