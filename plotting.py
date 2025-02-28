import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

matplotlib.use('Agg')  # To avoid error: _tkinter.TclError: no display name and no $DISPLAY environment variable


def draw_bounding_boxes(ax, particle_locations, image_height, image_width, box_width, box_height, box_color):
    for coords in particle_locations:
        x = coords[0] - box_width / 2
        y = image_height - coords[1] - box_height / 2

        box = patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(box)


def draw_bounding_circles(ax, particle_locations, image_height, image_width, circle_radius, box_color):
    for coords in particle_locations:
        x = coords[0]
        y = image_height - coords[1]

        circle = patches.Circle((x, y), circle_radius, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(circle)


def image_checking(image_tensor, result_dir):
    """
    Checking that is used in plotting functions
    :param image_tensor: PyTorch tensor of shape [C x H x W]
    :param result_dir: directory the plotting function will tru to save to
    """
    if not os.path.exists(result_dir):
        raise Exception("The folder you are trying to save to doesn't exist.")
    if len(image_tensor.shape) < 3:
        raise Exception("The image has no channel dimension")
    if image_tensor.shape[0] != 3:
        raise Exception("The image tensor must have three channels")


def save_image_with_bounding_object(image_tensor, particle_locations, object_type, object_parameters, result_dir,
                                     file_name, circle_color='r', z_threshold=9999):
    """
    Show the image and draw bounding boxes around the given pixel coordinates.

    :param image_tensor: Pytorch tensor representing the image with shape (C, H, W)
    :param particle_locations: Pytorch tensor of shape N x 2 (X and Y coords) or N x 3 (X, Y, and Z coords)
    :param object_type: String with type of object to draw, accepted: circle, box
    :param object_parameters: Dict with parameters for object_type, accepted: for circle: {circle_radius}, for box:
    {box_width, box_height}
    :param result_dir: Result directory to save images to
    :param file_name: File name of file that is saved
    :param circle_color: Border color of drawn circles
    :param z_threshold: Threshold for z value to decide if a circle should be plotted
    """
    # Validate object_parameters
    if object_type == "circle":
        required_params = ["circle_radius"]
    elif object_type == "box":
        required_params = ["box_width", "box_height"]
    else:
        raise Exception(f"Specified object type {object_type} is not supported.")
    for param in required_params:
        if param not in object_parameters:
            raise ValueError(f"Missing required parameter '{param}' for object type '{object_type}'.")

    image_checking(image_tensor, result_dir)

    channels, image_height, image_width = image_tensor.shape
    # Permute the tensor if necessary because imshow wants width x height x channels
    image_tensor = image_tensor.permute(1, 2, 0)
    fig, ax = plt.subplots(1)
    ax.imshow(image_tensor.cpu().numpy())

    if object_type == "circle":
        circle_radius = object_parameters["circle_radius"]
        draw_bounding_circles(ax, particle_locations, image_height, image_width, circle_radius, circle_color)
    elif object_type == "box":
        box_width = object_parameters["box_width"]
        box_height = object_parameters["box_height"]
        draw_bounding_boxes(ax, particle_locations, image_height, image_width, box_width, box_height, circle_color)

    plt.savefig(os.path.join(result_dir, file_name))
    plt.close(fig)


def compare_predictions_with_ground_truth(image_tensor, ground_truth, predictions, circle_radius, result_dir,
                                          file_name, gt_color='r', pred_color='r', z_threshold=9999):
    """
    Visualize model predictions side by side with the ground truth.

    :param image_tensor: Pytorch tensor representing the image with shape (C, H, W)
    :param ground_truth: Pytorch tensor of shape N x 2 (X and Y coords) or N x 3 (X, Y, and Z coords)
    :param predictions: Pytorch tensor of shape N x 2 (X and Y coords) or N x 3 (X, Y, and Z coords)
    :param circle_radius: Size of circle drawn around particles
    :param result_dir: Result directory to save images to
    :param file_name: File name of file that is saved
    :param gt_color: Border color of ground truth circles
    :param pred_color: Border color of prediction circles
    :param z_threshold: Threshold for z value to decide if a circle should be plotted
    """
    image_checking(image_tensor, result_dir)

    channels, image_height, image_width = image_tensor.shape
    # Permute the tensor if necessary because imshow wants width x height x channels
    image_tensor = image_tensor.permute(1, 2, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_tensor)
    ax2.imshow(image_tensor)
    ax1.set_title('Ground Truth')
    ax2.set_title('Predictions')

    draw_bounding_circles(ax1, ground_truth, image_height, image_width, circle_radius, gt_color)
    draw_bounding_circles(ax2, predictions, image_height, image_width, circle_radius, pred_color)

    plt.savefig(os.path.join(result_dir, file_name))
