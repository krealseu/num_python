import numpy as np

from .layer import ILayer

__all__ = ['FullConnection', 'Conv2D', 'Flatten', 'MaxPooling','Full']


def rand(m, n): return np.random.randn(m, n)


def _conv_kernel(data, filters, bias, result, stride1, stride2):
    n2, d2, w2, h2 = filters.shape
    n3, d3, w3, h3 = result.shape
    one_filter = filters.reshape(n2, -1).T
    for ww in range(w3):
        for hh in range(h3):
            d = (data[:, :, ww * stride1:ww * stride1 + w2, hh * stride2:hh * stride2 + h2]).reshape(n3, -1)
            result[:, :, ww, hh] = d @ one_filter + bias
    return


# @jit(nopython=True)
def _conv_kernel1(data, K, b, rs, s1, s2):
    n2, d2, w2, h2 = K.shape
    n3, d3, w3, h3 = rs.shape
    for nn in range(n3):
        for dd in range(d3):
            for ww in range(w3):
                for hh in range(h3):
                    d = data[nn, :, ww * s1:ww * s1 + w2, hh * s2:hh * s2 + h2]
                    kn = K[dd]
                    rs[nn, dd, ww, hh] = (d * kn).sum()
    return


# @jit(nopython=True)
def _conv_kernel2(data, K, b, rs, s1, s2):
    n2, d2, w2, h2 = K.shape
    n3, d3, w3, h3 = rs.shape
    for nn in range(n3):
        for dd in range(d3):
            for ww in range(w3):
                for hh in range(h3):
                    rs[nn, dd, ww, hh] = 0
                    for kd in range(d2):
                        for kw in range(w2):
                            for kh in range(h2):
                                rs[nn, dd, ww, hh] += K[dd, kd, kw, kh] * data[nn, kd, ww * s1 + kw, hh * s2 + kh]
    return


def _conv(data, filters, bias, strides, pad, conv_kernel):
    """
    :param data: shape (n,d,w,h)
    :param filters:shape (n,d,w,h)
    """
    n1, d1, w1, h1 = data.shape
    n2, d2, w2, h2 = filters.shape
    w3 = (w1 - w2 + 2 * pad[0]) / strides[0] + 1
    h3 = (h1 - h2 + 2 * pad[1]) / strides[0] + 1
    assert w3 % 1 == 0 and h3 % 1 == 0 and d1 == d2
    n3, d3, w3, h3 = n1, n2, int(w3), int(h3)
    result = np.empty((n3, d3, w3, h3))
    pad_data = np.pad(data, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), 'constant')
    conv_kernel(pad_data, filters, bias, result, strides[0], strides[1])
    return result


def conv(data, filters, bias, strider, pad, f): return _conv(data, filters, bias, (strider, strider), (pad, pad), f)


def _conv_gradient(x, filters, diff, strides, pad):
    n1, d1, w1, h1 = x.shape
    n2, d2, w2, h2 = filters.shape
    w3 = (w1 - w2 + 2 * pad[0]) / strides[0] + 1
    h3 = (h1 - h2 + 2 * pad[1]) / strides[0] + 1
    assert w3 % 1 == 0 and h3 % 1 == 0 and d1 == d2
    assert (n1, n2, int(w3), int(h3)) == diff.shape
    n3, d3, w3, h3 = diff.shape
    pad_data = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), 'constant')
    delta_filters = np.zeros_like(filters)
    delta_bias = np.zeros((1, n2))
    delta_filters_tmp = delta_filters.reshape(n2, -1)
    s1, s2 = strides
    for ww in range(w3):
        for hh in range(h3):
            d = (pad_data[:, :, ww * s1:ww * s1 + w2, hh * s2:hh * s2 + h2]).reshape(n3, -1)
            d_diff = diff[:, :, ww, hh]
            delta_filters_tmp += d_diff.T @ d
            delta_bias += d_diff.sum(axis=0, keepdims=True)
    return delta_filters, delta_bias


def conv_gradient(x, filters, diff, stride, pad): return _conv_gradient(x, filters, diff, (stride, stride), (pad, pad))


class Optimize:
    def __init__(self, beta1=0.9, beta2=0.995, lr=1.):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.m = 0
        self.v = 0

    @staticmethod
    def create():
        return Optimize(beta1=0.9, beta2=0.995, lr=0.01)

    def update(self, dw):
        self.m = self.m * self.beta1 + dw * (1 - self.beta1)
        self.v = self.v * self.beta2 + dw ** 2 * (1 - self.beta2)
        return self.lr * self.m / (np.sqrt(self.v) + self.lr)

    pass


class FullConnection(ILayer):
    """
    全连接层，处理两个神经层的连接
    """

    def __init__(self, i: int, o: int):
        super().__init__()
        self.dim = [i, o]
        self.w = rand(i, o) / np.sqrt(i)
        self.b = rand(1, o) / np.sqrt(i)
        self.x, self.y = np.zeros([1, i]), np.zeros([1, o])
        self._tmp_diff: np.ndarray = 0
        self.optimizeW = Optimize.create()
        self.optimizeB = Optimize.create()

    def predict(self, x: np.ndarray):
        return x @ self.w + self.b

    def forward(self, x: np.ndarray):
        assert x.shape[1] == self.dim[0]  # check input data
        self.x, self.y = x, x @ self.w + self.b
        return self.y

    def backward(self, diff: np.ndarray, back_diff=True):
        assert diff.shape == self.y.shape
        self._tmp_diff = diff
        if back_diff:
            return diff @ self.w.T
        else:
            return diff

    def update(self, lr=1):
        # change weight and bias
        diff = self._tmp_diff
        regularization = 1e-5
        dw = self.x.T @ diff + self.w * regularization
        db = diff.sum(axis=0, keepdims=True) + self.b * regularization
        self.w -= self.optimizeW.update(dw)
        self.b -= self.optimizeB.update(db)

    def params(self):
        pass

    pass


def sigmoid(x): return 1. / (1 + np.exp(-x))


def sigmoid_back(y, diff): return diff * y * (1 - y)

def prelu(x, r=0.01): return np.minimum(np.maximum(x * r, x),x*r+(1-r))

def prelu_back(y, diff, r=0.01): 
    tmp = np.where(y > 0, diff, diff * r)
    tmp = np.where(y < 1, tmp, tmp * r)
    return tmp

ff = sigmoid

bb = sigmoid_back


# ff = prelu

# bb = prelu_back

class Full(ILayer):
    """
    全连接层，处理两个神经层的连接
    """

    def __init__(self, i: int):
        super().__init__()
        self.dim = [i, i]
        self.w = np.zeros([i,i])
        self.b = np.zeros([1, i])
        self.x, self.y = np.zeros([1, i]), np.zeros([1, i])
        self._tmp_diff: np.ndarray = 0
        self.optimizeW = Optimize()
        self.optimizeB = Optimize.create()

    def predict(self, x: np.ndarray):
        return x * ff(x @ self.w + self.b)

    def forward(self, x: np.ndarray):
        assert x.shape[1] == self.dim[0]  # check input data
        self._y = x @ self.w + self.b
        self._hy = ff (self._y)
        self.x, self.y = x, x * self._hy
        return self.y

    def backward(self, diff: np.ndarray, back_diff=True):
        assert diff.shape == self.y.shape
        self._tmp_diff = bb(self._hy , diff * self.x)
        if back_diff:
            return self._tmp_diff @ self.w.T + diff * self._hy
        else:
            return diff

    def update(self, lr=1):
        # change weight and bias
        #return
        diff = self._tmp_diff
        regularization = 1e-5
        dw = self.x.T @ diff + self.w * regularization
        db = diff.sum(axis=0, keepdims=True) + self.b * regularization
        self.w -= self.optimizeW.update(dw)
        self.b -= self.optimizeB.update(db)

    def params(self):
        pass

    pass



class Conv2D(ILayer):
    """
    全连接层，处理两个神经层的连接
    """

    def __init__(self, shape):
        super().__init__()
        self.w = np.random.randn(*shape) / np.sqrt(np.cumprod(shape[1:])[-1])
        self.b = 0  # np.random.randn(shape) / np.sqrt(np.sum(shape[1:]))
        self.optimizeW = Optimize.create()
        self.optimizeB = Optimize.create()

    def predict(self, x: np.ndarray): return conv(x, self.w, 0, 1, 1, _conv_kernel)

    def forward(self, x: np.ndarray):
        self.x, self.y = x, conv(x, self.w, 0, 1, 1, _conv_kernel)
        return self.y

    def backward(self, diff: np.ndarray, back_diff=True):
        tmp_filters = np.flip(np.flip(self.w.swapaxes(0, 1), 2), 3)
        result = conv(diff, tmp_filters, 0, 1, 1, _conv_kernel)
        dw, _ = conv_gradient(self.x, self.w, diff, 1, 1)
        self.w += -dw * 0.001
        return result

    def update(self, lr=1):
        return

    pass


class MaxPooling(ILayer):
    def __init__(self, win_shape=(1, 2, 2), strides=(2, 2), pad=(0, 0)):
        super().__init__()
        self.win_shape = win_shape
        self.strides = strides
        self.padding = pad
        self._tmp_maxMask = None
        return

    def forward(self, x: np.ndarray):
        n1, d1, w1, h1 = x.shape
        d2, w2, h2 = self.win_shape
        w3 = (w1 - w2 + 2 * self.padding[0]) / self.strides[0] + 1
        h3 = (h1 - h2 + 2 * self.padding[1]) / self.strides[0] + 1
        assert w3 % 1 == 0 and h3 % 1 == 0 and (d1 / d2) % 1 == 0
        n3, d3, w3, h3 = n1, int(d1 / d2), int(w3), int(h3)
        result = np.empty((n3, d3, w3, h3))
        self._tmp_maxMask = np.zeros_like(x)
        pad_data = np.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                          'constant')
        for dd in range(d3):
            for ww in range(w3):
                for hh in range(h3):
                    win = pad_data[:, dd:dd + self.win_shape[0], ww:ww + self.win_shape[1], hh:hh + self.win_shape[2]]
                    result[:, dd, ww, hh] = np.max(win.reshape(n3, -1), axis=1)
                    self._tmp_maxMask[:, dd:dd + self.win_shape[0], ww:ww + self.win_shape[1],
                    hh:hh + self.win_shape[2]] = 1
        return result

    pass


class Flatten(ILayer):

    def __init__(self):
        super().__init__()
        self.x_shape = None
        return

    def predict(self, x: np.ndarray): return x.reshape(x.shape[0], -1)

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, diff: np.ndarray, back_diff=True):
        return diff.reshape(*self.x_shape)

    pass
