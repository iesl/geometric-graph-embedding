import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from PIL import Image

import time

def plot_2d_tbox(box_collection, negative_sampler, lr, negative_sampling_strategy=None):
    """

    Args:
        box_collection: (num_epochs, num_boxes, 2 (min/-max), 2 (x/y))

    Returns:

    """

    mins = box_collection[..., 0, :]        # (num_epochs, num_boxes, 2(x/y))
    # reverse -mu_z to mu_Z (c.f. TBox)
    maxs = -box_collection[..., 1, :]       # (num_epochs, num_boxes, 2(x/y))

    global_min, _ = mins.view(mins.shape[0] * mins.shape[1], 2).min(axis=0)
    global_max, _ = maxs.view(maxs.shape[0] * maxs.shape[1], 2).max(axis=0)

    epochs = mins.shape[0]

    filenames = []
    for e in range(epochs):     # iterate over epochs

        rectangles = dict()
        for i in range(mins.shape[1]):      # iterate over boxes in model

            xy = (mins[e, i, 0].item(), mins[e, i, 1].item())
            width = maxs[e, i, 0].item() - mins[e, i, 0].item()
            height = maxs[e, i, 1].item() - mins[e, i, 1].item()

            rectangle = Rectangle(xy=xy,
                                  width=width,
                                  height=height,
                                  angle=0,
                                  rotation_point='xy',
                                  fill=True,
                                  linewidth=0.1,
                                  mouseover=True,
                                  visible=True,
                                  edgecolor="r",
                                  facecolor="b",
                                  zorder=100,
                                  alpha=0.2)
            rectangles[i] = rectangle

        fig, ax = plt.subplots()
        ax.add_collection(PatchCollection(list(rectangles.values()), match_original=True))

        # annotate with node id
        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx #+ rectangles[r].get_width() / 2.0
            cy = ry #+ rectangles[r].get_height() / 2.0
            ax.annotate(r, (cx, cy), color='k', weight='bold',
                        fontsize=6, ha='center', va='center')

        plt.xlim([global_min[0].item(), global_max[0].item()])
        plt.ylim([global_min[1].item(), global_max[1].item()])
        plt.title(f"Boxes at epoch {e}")
        filename = f"/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/epoch-{e}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.clf()

    time.sleep(3.5)

    frames = [Image.open(fn) for fn in filenames]
    frame_one = frames[0]
    negative_sampling_strategy_str = f".strategy_{negative_sampling_strategy}" if negative_sampling_strategy else ""
    frame_one.save(f"/Users/brozonoyer/Desktop/IESL/box-training-methods/gifs/tbox.{negative_sampler}{negative_sampling_strategy_str}.{str(lr)}_lr.{str(epochs)}_epochs.gif",
                   format="GIF", append_images=frames, save_all=True, duration=150, loop=0)
