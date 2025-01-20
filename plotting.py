import matplotlib
matplotlib.use('Agg')  # Use this to avoid the error: _tkinter.TclError: no display name and no $DISPLAY environment variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_image_with_bounding_boxes(image_tensor, coordinates, size):
    """
    Show the image and draw bounding boxes around the given pixel coordinates.

    Parameters:
    - image_tensor: Tensor representing the image.
    - coordinates: NumPy array of shape (N, 2) representing pixel coordinates.
    - size: Size of the bounding box.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image_tensor)

    for (x, y) in coordinates:
        rect = patches.Rectangle((x - size // 2, y - size // 2), size, size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("example.png")


def main():
    # Image with bounding boxes example
    image_tensor = np.ones((100, 100))
    pixel_coordinates = np.array([[20, 20], [50, 50], [80, 80]])
    size = 10
    show_image_with_bounding_boxes(image_tensor, pixel_coordinates, size)


if __name__ == "__main__":
    main()
