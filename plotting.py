import matplotlib
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local imports

from util.utils import create_folder_if_missing

matplotlib.use('Agg')  # To avoid error: _tkinter.TclError: no display name and no $DISPLAY environment variable


def plot_loss_log(loss_log_path, result_dir):
    """
    Reads the loss log file and plots the losses as a semilogy plot.

    :param loss_log_path: Path to the loss log file. It needs to have a column "epoch" and a column "average_loss"
    :param result_dir: Directory to save the plot.
    """
    if os.path.exists(loss_log_path):
        # Read the loss log file
        loss_data = pd.read_csv(loss_log_path)
        epochs = loss_data['epoch']
        losses = loss_data['average_loss']

        # Plot the losses
        plt.figure()
        plt.semilogy(epochs, losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'losses_plot.png'))
        plt.close()
    else:
        print(f"Loss log file not found at {loss_log_path}")



def compare_images(image1, image2, file_name, output_location, title1, title2):
    """
    Plot two images side by side for comparison and save to file.

    :param image1: First image as a torch tensor with shape C, H, W
    :param image2: Second image as a torch tensor with shape C, H, W
    :param file_name: Name of the file to save the plot
    :param output_location: Directory to save the plot
    :param title1
    :param title2
    """
    image1 = image1.permute(1, 2, 0).cpu().numpy()
    image2 = image2.permute(1, 2, 0).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(image1)
    ax1.set_title(title1)
    ax1.axis('off')

    ax2.imshow(image2)
    ax2.set_title(title2)
    ax2.axis('off')

    create_folder_if_missing(output_location)

    plt.savefig(os.path.join(output_location, file_name))
    plt.close(fig)


def draw_output_boxes(ax, particle_locations, image_height, image_width, box_color):
    for coords in particle_locations:
        box_width = coords[2]
        box_height = coords[3]

        x = coords[0] - box_width / 2
        y = image_height - coords[1] - box_height / 2

        box = patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor=box_color, facecolor='none')
        ax.add_patch(box)


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


def save_image_with_bounding_object(image_tensor, particle_locations, object_type, object_parameters, result_dir,
                                    file_name, circle_color='r'):
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
    """
    if not os.path.exists(result_dir):
        raise Exception("The folder you are trying to save to doesn't exist.")

    fig, ax = plt.subplots(1)

    draw_image_with_objects_on_ax(ax, image_tensor, particle_locations, object_type, object_parameters, circle_color)

    plt.savefig(os.path.join(result_dir, file_name))
    plt.close(fig)


def draw_image_with_objects_on_ax(ax, image_tensor, particle_locations, object_type, object_parameters, edge_color):
    """
    Draws specified objects (circles or boxes) on the given axis.

    :param ax: Matplotlib axis to draw on
    :param image_tensor: Pytorch tensor representing the image with shape (C, H, W)
    :param particle_locations: Pytorch tensor of shape N x 2 (X and Y coords) or N x 4 (X, Y, Width, Height)
    :param object_type: String with type of object to draw, accepted: circle, box
    :param object_parameters: Dict with parameters for object_type, accepted: for circle: {circle_radius}, for box:
    {box_width, box_height}
    :param edge_color: Border color of drawn circles
    """
    # Validate object_parameters
    if object_type == "circle":
        required_params = ["circle_radius"]
    elif object_type == "box":
        required_params = ["box_width", "box_height"]
    elif object_type == "output_box":
        required_params = []
        if particle_locations.shape[1] != 4:
            raise Exception(f"The passed particle locations don't have enough information, we need "
                            f"[x, y, width, height] for each particle")
    else:
        raise Exception(f"Specified object type {object_type} is not supported.")
    for param in required_params:
        if param not in object_parameters:
            raise ValueError(f"Missing required parameter '{param}' for object type '{object_type}'.")

    # Validate image
    if len(image_tensor.shape) < 3:
        raise Exception("The image doesn't have enough dimensions, expected: [C, H, W]")
    if image_tensor.shape[0] != 3:
        raise Exception("The image tensor must have three channels")
    channels, image_height, image_width = image_tensor.shape
    # Permute the tensor if necessary because imshow wants width x height x channels
    image_tensor = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Draw
    ax.imshow(image_tensor)
    if object_type == "circle":
        circle_radius = object_parameters["circle_radius"]
        draw_bounding_circles(ax, particle_locations, image_height, image_width, circle_radius, edge_color)
    elif object_type == "box":
        box_width = object_parameters["box_width"]
        box_height = object_parameters["box_height"]
        draw_bounding_boxes(ax, particle_locations, image_height, image_width, box_width, box_height, edge_color)
    elif object_type == "output_box":
        draw_output_boxes(ax, particle_locations, image_height, image_width, edge_color)


def compare_predictions_with_ground_truth(image_tensor, ground_truth, predictions, object_type, object_parameters,
                                          result_dir, file_name, gt_color='r', pred_color='r'):
    """
    Visualize model predictions side by side with the ground truth.

    :param image_tensor: Pytorch tensor representing the image with shape (C, H, W)
    :param ground_truth: Pytorch tensor of shape N x 2 (X and Y coords) or N x 3 (X, Y, and Z coords)
    :param predictions: Pytorch tensor of shape N x 2 (X and Y coords) or N x 3 (X, Y, and Z coords)
    :param object_type: String with type of object to draw, accepted: circle, box
    :param object_parameters: Dict with parameters for object_type, accepted: for circle: {circle_radius}, for box:
    {box_width, box_height}
    :param result_dir: Result directory to save images to
    :param file_name: File name of file that is saved
    :param gt_color: Border color of ground truth circles
    :param pred_color: Border color of prediction circles
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)

    draw_image_with_objects_on_ax(ax1, image_tensor, ground_truth, object_type, object_parameters, gt_color)
    draw_image_with_objects_on_ax(ax2, image_tensor, predictions, "output_box", object_parameters, pred_color)

    ax1.set_title('Ground Truth')
    ax2.set_title('Predictions')

    plt.savefig(os.path.join(result_dir, file_name))
