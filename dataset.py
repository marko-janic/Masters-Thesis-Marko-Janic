import os

import mrcfile as mrc
import numpy as np

from torch.utils.data import Dataset
from numpy.lib.stride_tricks import as_strided


def create_sub_micrographs(micrograph, crop_size, sampling_points):
    height, width = micrograph.shape
    print(f"Height: {height}")
    print(f"Width: {width}")

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"

    # Calculate step size based on number of points per side
    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    print(f"step_size_x: {step_size_x}")
    print(f"step_size_y: {step_size_y}")

    sub_micrographs = []

    for i in range(sampling_points):  # horizontal steps
        for j in range(sampling_points):  # vertical steps
            start_x = i * step_size_x
            start_y = j * step_size_y
            end_x = start_x + crop_size
            end_y = start_y + crop_size

            # Ensure we don't go out of bounds
            if end_x <= width and end_y <= height:
                sub_micrographs.append(micrograph[start_x:end_x, start_y:end_y])

    return sub_micrographs


class ShrecDataset(Dataset):
    num_models = 10  # See shrec dataset
    projection_number = 29  # Which projection to use out of the 61 available. See alignment_simulated.txt files
    vit_input_size = 224  # The size we want our micrographs to be
    micrograph_size = 1024  # See shrec dataset
    num_crops_side = micrograph_size // vit_input_size  # We divide the micrograph into "sub-micrographs"
    num_sub_micrographs = num_crops_side * num_crops_side

    def __init__(self):
        self.data = []

        for i in range(self.num_models):
            with mrc.open(f'dataset/shrec21_full_dataset/model_{i}/projections_unbinned.mrc',
                          permissive=True) as f:
                micrograph = f.data[29]
                sub_micrographs = create_sub_micrographs(micrograph, self.vit_input_size)

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None
