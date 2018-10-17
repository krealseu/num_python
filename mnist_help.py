import struct

# import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    def load_labels(path):
        with open(path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            _labels = np.fromfile(lbpath, dtype=np.uint8)
        return _labels

    def load_images(path):
        with open(path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            _images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, rows * cols)
        return _images

    __labels = load_labels('data/train-labels.idx1-ubyte')
    __images = load_images('data/train-images.idx3-ubyte') / 255

    __test_labels = load_labels('data/t10k-labels.idx1-ubyte')
    __test_images = load_images('data/t10k-images.idx3-ubyte') / 255

    return __images, __labels, __test_images, __test_labels


# def showmnist(image):
#     image = image.reshape(28, 28)
#     plt.imshow(image, cmap='Greys_r')  # 显示图片
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()
#     return
