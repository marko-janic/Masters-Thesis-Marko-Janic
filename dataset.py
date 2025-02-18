import os

import mrcfile as mrc
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


def get_particle_locations_from_coordinates(coordinates_tl, sub_micrograph_size, particle_locations, particle_width,
                                            particle_height, orientation="normal"):
    """
    Given coordinates, this function determines the location of all relevant particles in the sub micrograph

    :param coordinates_tl: A tensor with coordinates of original micrograph of the top left most point in the
    sub micrograph. Index 0 is the x value and index 1 is the y value.
    :param sub_micrograph_size: Size of the sub micrographs, usually 224
    :param particle_locations: Pandas DataFrame containing all particle locations of the micrograph
    :param particle_height: Height of particle in micrograph
    :param particle_width: Width of particle in micrograph
    :param orientation: TODO
    :return: Pandas DataFrame with columns ['X', 'Y', 'Z', 'particle_width', 'particle_height'] which corresponds to
        the particles present in the sub micrograph determined by coordinate_tl
    """
    if orientation == "normal":
        x_min = coordinates_tl[0].item()
        x_max = x_min + sub_micrograph_size
        y_min = coordinates_tl[1].item()
        y_max = y_min + sub_micrograph_size

        selected_particles = particle_locations[(particle_locations['X'] >= x_min) &
                                                (particle_locations['X'] <= x_max) &
                                                (particle_locations['Y'] >= y_min) &
                                                (particle_locations['Y'] <= y_max)]
        # We subtract the minimum coordinates since we want the locations in the sub_micrograph so to speak
        selected_particles.loc[:, 'X'] = selected_particles['X'] - x_min
        selected_particles.loc[:, 'Y'] = selected_particles['Y'] - y_min

        # Add particle width and height information
        particle_widths = pd.Series([particle_width] * len(selected_particles))
        particle_heights = pd.Series([particle_height] * len(selected_particles))
        selected_particles = selected_particles.assign(particle_width=particle_widths)
        selected_particles = selected_particles.assign(particle_height=particle_heights)

        return selected_particles
    else:
        raise Exception(f'The orientation {orientation} is not a valid orientation')


def create_sub_micrographs(micrograph, crop_size, sampling_points):
    """
    Creates sub micrographs of the given micrograph by sliding a window of size crop_size x crop_size across the image
    from the top left to the bottom right. The total number of sub micrographs will be samping_points * sampling_points

    :param micrograph: A tensor of the micrograph to take sub micrographs from.
    :param crop_size: size of the sliding window
    :param sampling_points: The amount of stops the sliding window takes on one side of the micrograph.
    :return: A dataframe where the first column is a tensor of sub micrograph and the second a tensor with
    the coordinates of the top left most point of that sub micrograph in the original picture.
    """
    height, width = micrograph.shape
    #print(f"Height: {height}")
    #print(f"Width: {width}")

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"

    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    #print(f"step_size_x: {step_size_x}")
    #print(f"step_size_y: {step_size_y}")

    sub_micrographs_list = []
    for i in range(sampling_points):  # horizontal steps
        for j in range(sampling_points):  # vertical steps
            start_x = i * step_size_x
            start_y = j * step_size_y
            end_y = start_y + crop_size
            end_x = start_x + crop_size

            # Ensure we don't go out of bounds
            if end_x <= width and end_y <= height:
                sub_micrographs_list.append((micrograph[start_y:end_y, start_x:end_x],
                                             torch.tensor([start_x, start_y])))

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "top_left_coordinates"])

    return sub_micrographs


class ShrecDataset(Dataset):
    projection_number = 29  # Which projection to use for noisy example. See alignment_simulated.txt files

    def __init__(self, sampling_points, micrograph_size=512, sub_micrograph_size=224,
                 dataset_path='dataset/shrec21_full_dataset/', model_number=1, particle_width=16, particle_height=16):
        """
        Dataset Loader for Shrec21 Dataset.

        :param sampling_points: Determines number of sub_micrographs, sampling_points^2 = number of sub micrographs
        :param micrograph_size: See shrec dataset, grandmodel is 512 x 512
        :param sub_micrograph_size: The size we want our sub micrographs to be
        :param dataset_path: Path to dataset
        :param particle_height: Height of particle in micrograph
        :param particle_width: Width of particle in micrograph
        """

        self.sub_micrograph_size = sub_micrograph_size
        self.model_number = model_number
        self.micrograph_size = micrograph_size
        self.sampling_points = sampling_points
        self.dataset_path = dataset_path
        self.particle_width = particle_width
        self.particle_height = particle_height

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        sep=r'\s+', names=columns).drop(columns=['rotation_Z1', 'rotation_X',
                                                                 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, f'model_{self.model_number}/grandmodel.mrc'),
                      permissive=True) as f:
            self.micrograph = torch.tensor(f.data.sum(axis=0))
            self.micrograph /= self.micrograph.max()  # Normalize the data between 0 and 1

        self.sub_micrographs = create_sub_micrographs(self.micrograph, self.sub_micrograph_size, self.sampling_points)

    def __len__(self):
        return len(self.sub_micrographs)

    def __getitem__(self, idx):
        """
        Returns two tensors, one with the sub micrograph and one with the coordinates.
        :param idx: The index to take from
        """
        sub_micrograph_entry = self.sub_micrographs.iloc[idx]
        # TODO: look at if we want to do the channels like this (just repetition)
        return (sub_micrograph_entry['sub_micrograph'].unsqueeze(0).repeat(3, 1, 1),
                sub_micrograph_entry['top_left_coordinates'])
