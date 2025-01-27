import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

matplotlib.use('Agg')  # To avoid error: _tkinter.TclError: no display name and no $DISPLAY environment variable


def save_image_with_bounding_boxes(image_tensor, particle_locations, box_size, result_dir, file_name, box_color='r'):
    """
    Show the image and draw bounding boxes around the given pixel coordinates.

    :param image_tensor: Tensor representing the image.
    :param particle_locations: Pandas Dataframe of shape (N, 2) representing pixel coordinates.
    :param box_size: Size of the bounding box.
    :param result_dir: Result directory to save to
    :param file_name: File name of file that is saved
    :param box_color: Border color of drawn boxes
    """
    if not os.path.exists(result_dir):
        raise Exception("The folder you are trying to save to doesn't exist.")

    fig, ax = plt.subplots(1)
    ax.imshow(image_tensor, cmap='gray')

    for _, row in particle_locations.iterrows():
        x = row['X']
        y = row['Y']
        rect = patches.Rectangle((x, y), box_size, box_size, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

    plt.savefig(os.path.join(result_dir, file_name))
