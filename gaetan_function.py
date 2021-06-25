import logging
from typing import NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import treefiles as tf
from matplotlib.animation import FuncAnimation


class Param(NamedTuple):
    name: str
    values: List[float]


def main(x: Param, y: Param):
    root = tf.Tree.new(__file__)
    root.file(vid="video.mp4")

    fig, axs = prepare_canvas(x, y, figsize=(6, 6))
    nx, ny = len(x.values), len(y.values)

    # Get dummy data
    dims, frames = (10, 10), 15
    videos = [get_random(dims, frames) for _ in axs.ravel()]

    imgs = [ax.imshow(np.zeros(dims)) for ax in axs.ravel()]
    set_labels(x, y, axs)

    def update(frame):
        for n in range(nx * ny):
            imgs[n].set_array(videos[n][..., frame])
        return imgs

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=300)
    ani.save(root.vid)

    plt.tight_layout()
    plt.show()


def set_labels(x: Param, y: Param, axs):
    for k in ["x", "y"]:
        for i, v in enumerate(locals()[k].values):
            ax = (axs if k == "x" else axs.T)[0, i]
            getattr(ax, f"set_{k}ticks")([sum(getattr(ax, f"get_{k}lim")()) / 2])
            getattr(ax, f"set_{k}ticklabels")([f"{v}"])


def prepare_canvas(x: Param, y: Param, title=None, **kw):
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True
    plt.rcParams["ytick.left"] = plt.rcParams["ytick.labelleft"] = True
    pltnot = dict(top=False, bottom=False, left=False, right=False)

    nx, ny = len(x.values), len(y.values)
    fig, axs = plt.subplots(nrows=ny, ncols=nx, sharex="col", sharey="row", **kw)

    for ii in axs.ravel():
        ii.tick_params(**pltnot)

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(**pltnot, labelcolor="none")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(x.name, fontsize=14)
    ax.set_ylabel(y.name, fontsize=14)

    if title is not None:
        fig.suptitle(title)

    return fig, axs


def get_random(dims, frames):
    vid = [*dims, 3, frames]
    return np.random.rand(*vid)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    _x = Param("$x_1$", [1, 2, 3])
    _y = Param("$x_2$", [5, 6, 7])

    main(_x, _y)