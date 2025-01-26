import os

import mrcfile as mrc
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def create_sub_micrographs(micrograph, crop_size, sampling_points):
    """
    Creates sub micrographs of the given micrograph by sliding a window of size crop_size x crop_size across the image
    from the top left to the bottom right. The total number of sub micrographs will be samping_points * sampling_points

    :param micrograph: the micrograph to take sub micrographs from.
    :param crop_size: size of the sliding window
    :param sampling_points: The amount of stops the sliding window takes on one side of the micrograph.
    :return: A dataframe where the first column is the sub micrograph and the second the coordinates of the top left
    most point of that sub micrograph in the original picture.
    """
    height, width = micrograph.shape
    print(f"Height: {height}")
    print(f"Width: {width}")

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"

    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    print(f"step_size_x: {step_size_x}")
    print(f"step_size_y: {step_size_y}")

    sub_micrographs_list = []
    for i in range(sampling_points):  # horizontal steps
        for j in range(sampling_points):  # vertical steps
            start_x = i * step_size_x
            start_y = j * step_size_y
            end_x = start_x + crop_size
            end_y = start_y + crop_size

            # Ensure we don't go out of bounds
            if end_x <= width and end_y <= height:
                sub_micrographs_list.append((micrograph[start_x:end_x, start_y:end_y], (start_x, start_y)))

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "top_left_coordinates"])

    return sub_micrographs


class ShrecDataset(Dataset):
    num_models = 10  # See shrec dataset
    model_number = 1  # Model to select for this iteration
    dataset_path = "dataset/shrec21_full_dataset/"
    projection_number = 29  # Which projection to use for noisy example. See alignment_simulated.txt files
    vit_input_size = 224  # The size we want our micrographs to be
    micrograph_size = 1024  # See shrec dataset
    num_crops_side = micrograph_size // vit_input_size  # We divide the micrograph into "sub-micrographs"
    num_sub_micrographs = num_crops_side * num_crops_side

    def __init__(self):
        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        delim_whitespace=True, names=columns).drop(columns=['Z', 'rotation_Z1',
                                                                            'rotation_X', 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, 'model_{i}/grandmodel.mrc'),
                      permissive=True) as f:
            self.micrograph = np.sum(f.data, axis=0)
        self.sub_micrographs = create_sub_micrographs(self.micrograph, self.vit_input_size, 9)

    def __len__(self):
        return len(self.sub_micrographs)

    def __getitem__(self, idx):
        return None
