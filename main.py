import logging
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import treefiles as tf
from matplotlib.animation import FuncAnimation

from patterns import BasePattern


class Game:
    def __init__(self, image: Union[BasePattern, np.ndarray]):
        if isinstance(image, BasePattern):
            image = image.pattern
        self.dims = image.shape
        self.total_image = image[..., np.newaxis]
        self.root = tf.Tree.new(__file__)
        self.deb: Callable = self.root.dir("debug").dump(clean=False).path

    def make_animation(self, fname: str):
        fig, ax = plt.subplots()  # figsize=(6, 2)
        ax.tick_params(bottom=False, left=False, labelcolor="none")
        im = ax.imshow(self.total_image[..., 0], cmap=plt.get_cmap("binary"))
        ax.set_yticks(np.arange(0, self.dims[0]) - 0.5)
        ax.set_xticks(np.arange(0, self.dims[1]) - 0.5)
        ax.grid(color="grey", alpha=0.5)

        # noinspection PyRedundantParentheses
        def update(frame):
            im.set_array(self.total_image[..., frame])
            return (im,)

        fig.tight_layout()
        ani = FuncAnimation(
            fig, update, frames=self.total_image.shape[-1], blit=True, interval=400
        )
        # plt.show()
        if fname is not None:
            ani.save(fname)
            log.info(f"Video saved to file://{fname}")

    def compute_generations(self, max_iter):
        it = 0
        while True:
            try:
                if it >= max_iter:
                    raise StopIteration(f"max_iter: {it}/{max_iter}")
                self.compute_generation()
                it += 1
            except StopIteration as e:
                log.info(f"StopIteration (iter {it}): {e}")
                return

    def compute_generation(self):
        """
        Add an image to the self.total_image stack
        """
        new_image = self.update_gol()

        if np.all(self.total_image[..., -1] == new_image):
            raise StopIteration("images are identical")

        new_image = new_image[..., np.newaxis]
        self.total_image = np.append(self.total_image, new_image, axis=-1)

    def update_gol(self):
        cur_image = self.total_image[..., -1].copy()
        sh = cur_image.shape
        container = np.zeros((sh[0] + 2, sh[1] + 2, *sh[2:]))
        container[1:-1, 1:-1] = cur_image
        for j in range(1, sh[1] - 1):
            for i in range(1, sh[0] - 1):
                mask = container[i - 1 : i + 2, j - 1 : j + 2, ...]
                cur_image[i - 1, j - 1, ...] = Game.get_fate(mask)
        return cur_image

    @staticmethod
    def get_fate(mask):
        if mask[1, 1] == 1:
            return 1 if 1 < mask.sum() - 1 < 4 else 0
        else:
            return 1 if mask.sum() > 2 else 0

    def get_ij(self, n):
        nx, ny = self.total_image.shape[:2]
        return n // ny + 1, n % nx + 1

    def plot_frame(self, frame_number):
        fig, ax = plt.subplots()
        ax.imshow(self.total_image[..., frame_number], cmap=plt.get_cmap("binary"))
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    from patterns import Blinker, Callahan5, Pulsar

    pats = {Blinker: (9, 9), Callahan5: (20, 20), Pulsar: (20, 20)}

    for k, v in pats.items():
        g = Game(k(v))
        g.compute_generations(max_iter=100)
        g.make_animation(g.deb(f"{k.__name__}.mp4"))
