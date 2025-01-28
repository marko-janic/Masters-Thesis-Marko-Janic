import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.transforms import Affine2D

matplotlib.use('Agg')  # To avoid error: _tkinter.TclError: no display name and no $DISPLAY environment variable


def save_image_with_bounding_boxes(image_tensor, particle_locations, circle_radius, result_dir,
                                   file_name, box_color='r'):
    """
    Show the image and draw bounding boxes around the given pixel coordinates.

    :param image_tensor: Tensor representing the image
    :param particle_locations: Pandas Dataframe of shape (N, 2) representing pixel coordinates.
    :param circle_radius: Size of circle drawn around particle
    :param result_dir: Result directory to save to
    :param file_name: File name of file that is saved
    :param box_color: Border color of drawn boxes
    """
    if not os.path.exists(result_dir):
        raise Exception("The folder you are trying to save to doesn't exist.")

    fig, ax = plt.subplots(1)

    ax.imshow(image_tensor, cmap='gray')

    for _, row in particle_locations.iterrows():
        box_color = 'b'
        if row['class'] == 'fiducial':
            box_color = 'r'
        if row['class'] == 'vesicle':
            box_color = 'g'

        x = row['X']
        y = row['Y']
        rect = patches.Circle((x, y), circle_radius, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

    plt.savefig(os.path.join(result_dir, file_name))
