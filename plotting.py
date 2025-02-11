import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.transforms import Affine2D

matplotlib.use('Agg')  # To avoid error: _tkinter.TclError: no display name and no $DISPLAY environment variable


def save_image_with_bounding_circles(image_tensor, particle_locations, circle_radius, result_dir,
                                   file_name, box_color='r', z_threshold=9999):
    """
    Show the image and draw bounding boxes around the given pixel coordinates.

    :param image_tensor: Pytorch tensor representing the image with shape (C, H, W) or (H, W)
    :param particle_locations: Pandas dataframe with columns ['X', 'Y', 'Z'] and rows equal to the number of
        particles in the image
    :param circle_radius: Size of circle drawn around particle
    :param result_dir: Result directory to save images to
    :param file_name: File name of file that is saved
    :param box_color: Border color of drawn circles
    :param z_threshold: Threshold for z value to decide if a circle should be plotted
    """
    if not os.path.exists(result_dir):
        raise Exception("The folder you are trying to save to doesn't exist.")

    fig, ax = plt.subplots(1)

    # Premute the tensor if necessary because imshow wants width x height x channels
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.permute(1, 2, 0)
    elif image_tensor.dim() < 2 or image_tensor.dim() > 3:
        raise Exception("The image tensor should have shape (C, H, W) or (H, W)")

    if image_tensor.dim() != 2:
        image_tensor = image_tensor[:, :, 0]
    ax.imshow(image_tensor)

    for _, row in particle_locations.iterrows():
        if row['Z'] > z_threshold:
            continue

        box_color = 'b'
        if row['class'] == 'fiducial':
            box_color = 'r'
        if row['class'] == 'vesicle':
            box_color = 'g'

        x = row['X']
        y = row['Y']
        circle = patches.Circle((x, y), circle_radius, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(circle)

    plt.savefig(os.path.join(result_dir, file_name))
