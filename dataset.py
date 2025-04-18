import os
import random
import torch

import mrcfile as mrc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

# Local imports
from util.utils import create_folder_if_missing


def create_dummy_dataset(image_size, num_images, min_particles, max_particles, particle_radius, output_dir,
                         max_overlap=0.0):
    """
    Creates a dummy dataset and saves it.

    :param image_size: Size of images that are created
    :param num_images: Number of images to create
    :param min_particles: Minimum number of particles per image
    :param max_particles: Maximum number of particles per image
    :param particle_radius: Size of particles that are drawn
    :param output_dir: Directory to save the dataset
    :param max_overlap: Maximum allowed percentage overlap between particles (0.0 to 1.0)
    """
    create_folder_if_missing(output_dir)
    create_folder_if_missing(os.path.join(output_dir, 'data'))

    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(f"image_size: {image_size}\n")
        f.write(f"num_images: {num_images}\n")
        f.write(f"min_particles: {min_particles}\n")
        f.write(f"max_particles: {max_particles}\n")
        f.write(f"particle_radius: {particle_radius}\n")
        f.write(f"output_dir: {output_dir}\n")

    for i in tqdm(range(num_images), desc="Creating images"):
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, image_size)
        ax.set_ylim(0, image_size)
        ax.axis('off')

        coordinates = []
        num_dots = random.randint(min_particles, max_particles)
        for _ in range(num_dots):
            counter = 0
            # Check 
            while True:
                x = np.random.randint(0, image_size)
                y = np.random.randint(0, image_size)
                new_circle = patches.Circle((x, y), radius=particle_radius, color='black')

                # Check overlap with existing particles
                overlap = False
                for coord in coordinates:
                    dist = np.sqrt((x - coord[0])**2 + (y - coord[1])**2)
                    if dist < 2 * particle_radius * (1 - max_overlap):
                        overlap = True
                        break

                if not overlap:
                    coordinates.append((x, y))
                    ax.add_patch(new_circle)
                    break

                if counter > 100:
                    print(f"Too many attempts to place a particle without overlap. Skipping one patch at image {i}")
                    break

        image_path = os.path.join(output_dir, f'data/micrograph_{i}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        coordinates_path = os.path.join(output_dir, f'data/micrograph_{i}_coords.txt')
        with open(coordinates_path, 'w') as f:
            for coord in coordinates:
                f.write(f'{coord[0]},{coord[1]}\n')


class DummyDataset(Dataset):
    def __init__(self, dataset_size, particle_width=10, particle_height=10, image_width=224, image_height=224,
                 dataset_path='dataset/dummy_dataset/data'):
        self.dataset_path = dataset_path
        self.particle_width = particle_width
        self.particle_height = particle_height
        self.image_width = image_width
        self.image_height = image_height

        self.micrographs = []
        self.targets = []

        preprocess = transforms.Compose([
            transforms.ToTensor(),  # Automatically rescales between 0 and 1
        ])

        for idx in range(dataset_size):

            # The image (micrograph)
            micrograph_path = os.path.join(self.dataset_path, f'micrograph_{idx}.png')
            if not os.path.isfile(micrograph_path):
                raise Exception(f"The file {micrograph_path} doesn't exist.")

            image = Image.open(micrograph_path).convert("RGB")
            self.micrographs.append(preprocess(image))

            # The target (coordinates + classes)
            coordinates_path = os.path.join(self.dataset_path, f'micrograph_{idx}_coords.txt')
            if not os.path.isfile(coordinates_path):
                raise Exception(f"The file {coordinates_path} doesn't exist.")
            coordinates = pd.read_csv(coordinates_path, sep=',', header=None, names=['X', 'Y'])

            if coordinates.empty:  # TODO: what do we do in this case?
                raise Exception(f"The file {coordinates_path} is empty.")

            coordinates = torch.tensor(coordinates.values, dtype=torch.float32)
            particle_sizes = torch.tensor([self.particle_width, self.particle_height],
                                          dtype=torch.float32).repeat(len(coordinates), 1)
            bboxes = (torch.cat((coordinates, particle_sizes), dim=1) /
                      torch.Tensor([self.image_width, self.image_height, self.image_width, self.image_height]))
            classes = torch.zeros(len(bboxes))  # Zeros since we have one class (particle)

            target = {
                "boxes": bboxes,
                "labels": classes,
                "orig_size": torch.tensor([self.image_width, self.image_height]),
                "image_id": idx
            }

            self.targets.append(target)

        self.micrographs = torch.stack(self.micrographs)

    def get_targets_from_target_indexes(self, indexes, device):
        targets = []
        for target_index in indexes:
            target = self.targets[target_index]
            # Move target to the same device as the model, take everything except image_id
            target = {k: v.to(device) for k, v in target.items() if k != "image_id"}
            targets.append(target)

        return targets

    def __len__(self):
        return len(self.micrographs)

    def __getitem__(self, idx):
        """
        Returns a (micrograph, index) pair
        :param idx: Index of where to take from
        :return: Tuple: (Micrograph tensor of size [3 x 224 x 224], idx). You can use idx to get the targets. We do this
        because the loss function expects a list of dicts and return the dict here will make one dict with batched
        tensors inside, which is now what we want
        """
        return self.micrographs[idx], idx


# TODO: this is super outdated and needs to be reworked
def get_particle_locations_from_coordinates(coordinates_tl, sub_micrograph_size, particle_locations,
                                            orientation="normal"):
    """
    Given coordinates, this function determines the location of all relevant particles in the sub micrograph

    :param coordinates_tl: A tensor with coordinates of original micrograph of the top left most point in the
    sub micrograph. Index 0 is the x value and index 1 is the y value.
    :param sub_micrograph_size: Size of the sub micrographs, usually 224
    :param particle_locations: Pandas DataFrame containing all particle locations of the micrograph
    :param orientation: TODO
    :return: Pandas DataFrame with columns ['X', 'Y', 'Z'] which corresponds to the particles present in the
        sub micrograph determined by coordinate_tl
    """
    if orientation == "normal":
        x_min = coordinates_tl[0].item()
        x_max = x_min + sub_micrograph_size
        y_min = coordinates_tl[1].item()
        y_max = y_min + sub_micrograph_size
        #print("x_min:", x_min)
        #print("x_max:", x_max)
        #print("y_min:", y_min)
        #print("y_max:", y_max)

        selected_particles = particle_locations[(particle_locations['X'] >= x_min) &
                                                (particle_locations['X'] <= x_max) &
                                                (particle_locations['Y'] >= y_min) &
                                                (particle_locations['Y'] <= y_max)]
        # We subtract the minimum coordinates since we want the locations in the sub_micrograph so to speak
        selected_particles.loc[:, 'X'] = selected_particles['X'] - x_min
        selected_particles.loc[:, 'Y'] = selected_particles['Y'] - y_min

        return selected_particles
    else:
        raise Exception(f'The orientation {orientation} is not a valid orientation')


# TODO: this is super outdated and needs to be reworked
def get_coordinates_in_sub_micrograph(coordinates_in_original_image, coordinate_tl):
    """
    Scales the coordinates from the original micrograph to the sub micrograph.

    :param coordinates_in_original_image: A tensor of coordinates in the original micrograph.
    :param coordinate_tl: The top left coordinate of the sub micrograph.
    :return: A tensor of coordinates in the sub micrograph.
    """
    x_min = coordinate_tl[0].item()
    y_min = coordinate_tl[1].item()

    scaled_coordinates = coordinates_in_original_image.clone()
    scaled_coordinates[:, 0] -= x_min
    scaled_coordinates[:, 1] -= y_min

    return scaled_coordinates


# TODO: this is super outdated and needs to be reworked
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


# TODO: this is super outdated and needs to be reworked
class ShrecDataset(Dataset):
    projection_number = 29  # Which projection to use for noisy example. See alignment_simulated.txt files

    def __init__(self, sampling_points, micrograph_size=512, sub_micrograph_size=224, model_number=1,
                 dataset_path='dataset/shrec21_full_dataset/'):
        """
        Dataset Loader for Shrec21 Dataset.

        :param sampling_points: Determines number of sub_micrographs, sampling_points^2 = number of sub micrographs
        :param micrograph_size: See shrec dataset
        :param sub_micrograph_size: The size we want our sub micrographs to be
        :param model_number: Model to select for this iteration
        :param dataset_path: Path to dataset
        """

        self.dataset_path = dataset_path
        self.model_number = model_number  # TODO: needed later for noisy projection
        self.micrograph_size = micrograph_size  # This is only needed for creating the sub micrographs

        self.sub_micrograph_size = sub_micrograph_size  #TODO: RENAME THIS TO
        self.sampling_points = sampling_points

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        sep=r'\s+', names=columns).drop(columns=['rotation_Z1', 'rotation_X',
                                                                 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, f'model_{self.model_number}/grandmodel.mrc'),
                      permissive=True) as f:
            self.micrograph = torch.tensor(f.data.sum(axis=0))  # We know 0 is correct from testing, 0 is top view
            # We need to normalize our micrograph between 0 and 1 for the loss function and the ViT model
            self.micrograph = self.micrograph/self.micrograph.max()

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


if __name__ == "__main__":
    create_dummy_dataset(224, 20000, 1, 5, 40,
                         "dataset/dummy_dataset_no_overlap", max_overlap=0.0)
