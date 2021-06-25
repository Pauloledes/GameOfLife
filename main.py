import logging

import matplotlib.pyplot as plt
import numpy as np
import treefiles as tf
from matplotlib.animation import FuncAnimation


class Game:
    def __init__(self, image):
        self.total_image = image[..., np.newaxis]
        self.root = tf.Tree.new(__file__)
        self.root.file(vid="gol.avi")
        self.deb = self.root.dir("debug").dump(clean=True).path

    def plot_frame(self, frame_number):
        fig, ax = plt.subplots()
        ax.imshow(self.total_image[..., frame_number], cmap=plt.get_cmap('binary'))
        plt.show()

    def animate(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.total_image[..., 0], cmap=plt.get_cmap("binary"))

        def update(frame):
            im.set_array(self.total_image[..., frame])
            return (im,)

        ani = FuncAnimation(fig, update, frames=10, blit=True, interval=300)
        ani.save(self.root.vid)

    def compute_generation(self):
        """
        Add an image to the self.total_image stack
        """
        new_image = np.random.randint(2, size=(20, 20, 1))
        new_image = new_image[..., np.newaxis]
        self.total_image = np.append(self.total_image, new_image, axis=-1)


def main():
    image = np.random.randint(2, size=(20, 20, 1))

    g = Game(image)
    g.compute_generation()
    # g.plot_frame(0)
    g.animate()
    # g.plot_frame(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()
    main()
