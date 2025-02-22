import os
import torch

# Local imports


def print_separator(label="", char="=", length=100):
    print()
    if label:
        side_length = (length - len(label) - 2) // 2
        print(f"{char * side_length} {label} {char * side_length}".center(length, char))
    else:
        print(char * length)
    print()


def create_folder_if_missing(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


def transform_coords_to_pixel_coords(image_width, image_height, coords):
    """
    Transforms coordinates from the range [0, 1] to pixel coordinates.

    Args:
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        coords (torch.Tensor): A tensor of shape [1, Batch, x, y] with coordinates in the range [0, 1].

    Returns:
        torch.Tensor: A tensor of the same shape as coords with pixel coordinates.
    """
    # Ensure coords is a tensor
    # if not isinstance(coords, torch.Tensor):
    #     raise TypeError("coords must be a torch.Tensor")

    # Scale x coordinates by image width and y coordinates by image height
    coords[:, 0] = coords[:, 0] * image_width
    coords[:, 1] = coords[:, 1] * image_height

    return coords
