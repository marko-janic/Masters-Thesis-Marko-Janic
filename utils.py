import os
import torch


def create_folder_if_missing(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


def transform_coords_to_pixel_coords(image_height, image_width, coords):
    """
    Transforms coordinates from the range [0, 1] to pixel coordinates.

    Args:
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        coords (torch.Tensor): A tensor of shape [batch, num_predictions, 4] or [num_predictions, 4] with coordinates
        in the range [0, 1].

    Returns:
        torch.Tensor: A tensor of the same shape as coords with pixel coordinates.
    """
    assert isinstance(coords, torch.Tensor), "Input 'coords' must be a torch.Tensor"
    assert coords.dim() == 3 and coords.size(2) == 4, \
        "Input tensor 'coords' must have shape [batch, num_predictions, 4]"

    # Scale coordinates
    coords[:, :, 0] *= image_height
    coords[:, :, 1] *= image_width
    coords[:, :, 2] *= image_width
    coords[:, :, 3] *= image_height

    return coords
