from enum import Enum

import numpy as np

__all__ = ['ILayer']


class Type(Enum):
    connection = 1
    function = 2
    classic = 3
    pass


class ILayer:
    def __init__(self, _type: Type = Type.classic):
        self._type = _type
        self.x = self.y = 0
        return

    def predict(self, x: np.ndarray): return self.forward(x)

    def forward(self, x: np.ndarray): return x

    def backward(self, diff: np.ndarray, back_diff=True): return diff

    def update(self, lr=1): return

    def params(self): return

    def clear(self):
        """
        清除训练过程中的变量，空出内存
        """
        return

    pass
