import logging
from typing import Tuple

import numpy as np
import treefiles as tf

TDims = Tuple[int, int]


class BasePattern:
    core_dims_ = None
    list_ = None

    def __init__(self, image):
        self.pattern = image

    @classmethod
    def from_list(cls, dims: TDims, pixels):
        arr = np.zeros(dims).ravel()
        arr[pixels] = 1
        arr = arr.reshape(dims)
        c = cls(image=arr)
        return c

    @staticmethod
    def _include(dims: TDims, pattern):
        s = pattern.shape
        image = np.zeros(dims)
        ixy = [image.shape[i] // 2 - s[i] // 2 for i in range(pattern.ndim)]
        image[ixy[0] : ixy[0] + s[0], ixy[1] : ixy[1] + s[1]] = pattern
        return image

    def init_core(self, dims: TDims):
        pat = BasePattern.from_list(self.core_dims_, self.list_).pattern
        return BasePattern._include(dims, pat)


class Callahan5(BasePattern):
    core_dims_ = (5, 5)
    list_ = [0, 1, 2, 4, 5, 13, 14, 16, 17, 19, 20, 22, 24]

    def __init__(self, dims: TDims):
        super().__init__(self.init_core(dims))


class Callahan1(BasePattern):
    core_dims_ = (1, 39)
    t__ = list(range(39))
    list_ = t__[:8] + t__[10:15] + t__[19:22] + t__[27:34] + t__[35:]

    def __init__(self, dims: TDims):
        super().__init__(self.init_core(dims))


class Blinker(BasePattern):
    core_dims_ = (3, 1)
    list_ = [0, 1, 2]

    def __init__(self, dims: TDims):
        super().__init__(self.init_core(dims))


class Pulsar(BasePattern):
    core_dims_ = (17, 17)

    def __init__(self, dims: TDims):
        arr = np.zeros(Pulsar.core_dims_)
        arr[2, 4:7] = 1
        arr[4:7, 7] = 1
        arr += arr.T
        arr += arr[:, ::-1]
        arr += arr[::-1, :]
        super().__init__(BasePattern._include(dims, arr))


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    image_ = Pulsar((30, 30))
    # print(image_.pattern)

    from main import Game

    Game(image_.pattern).plot_frame(0)
