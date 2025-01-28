import os

import mrcfile as mrc
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def get_particle_locations_from_coordinates(coordinates_tl, crop_size, particle_locations, orientation="normal"):
    """
    Given coordinates, this function determines the location of all relevant particles in the sub micrograph

    :param coordinates_tl: Coordinates of original micrograph of the top left most point in the sub micrograph
    :param crop_size: Size of the sub micrographs, usually 224
    :param particle_locations: Pandas DataFrame containing all particle locations of the micrograph
    :param orientation: TODO
    :return: Pandas DataFrame with the particles present in the sub micrograph
    """
    if orientation == "normal":
        x_min = coordinates_tl[0]
        x_max = x_min + crop_size
        y_min = coordinates_tl[1]
        y_max = y_min + crop_size
        print("x_min:", x_min)
        print("x_max:", x_max)
        print("y_min:", y_min)
        print("y_max:", y_max)

        selected_particles = particle_locations[(particle_locations['X'] >= x_min) &
                                                (particle_locations['X'] <= x_max) &
                                                (particle_locations['Y'] >= y_min) &
                                                (particle_locations['Y'] <= y_max)]
        # We subtract the minimum coordinates since we want the locations in the sub_micrograph so to speak
        selected_particles['X'] = selected_particles['X'] - x_min
        selected_particles['Y'] = selected_particles['Y'] - y_min

        return selected_particles
    else:
        raise Exception(f'The orientation {orientation} is not a valid orientation')


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
                sub_micrographs_list.append((micrograph[start_y:end_y, start_x:end_x], (start_x, start_y)))

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "top_left_coordinates"])

    return sub_micrographs


class ShrecDataset(Dataset):
    num_models = 10  # See shrec dataset
    model_number = 1  # Model to select for this iteration
    dataset_path = "dataset/shrec21_full_dataset/"
    projection_number = 29  # Which projection to use for noisy example. See alignment_simulated.txt files
    sub_micrograph_size = 150  # The size we want our micrographs to be
    micrograph_size = 512  # See shrec dataset
    sampling_points = 2  # Determines number of sub_micrographs

    def __init__(self, sampling_points, dataset_path="dataset/shrec21_full_dataset/"):
        self.sampling_points = sampling_points
        self.dataset_path = dataset_path

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        sep=r'\s+', names=columns).drop(columns=['Z', 'rotation_Z1', 'rotation_X',
                                                                 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, f'model_{self.model_number}/grandmodel.mrc'),
                      permissive=True) as f:
            self.micrograph = np.sum(f.data, axis=0)

        self.sub_micrographs = create_sub_micrographs(self.micrograph, self.sub_micrograph_size, self.sampling_points)

    def __len__(self):
        return len(self.sub_micrographs)

    def __getitem__(self, idx):
        return self.sub_micrographs.iloc[idx]
