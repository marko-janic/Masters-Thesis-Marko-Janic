import argparse
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--min_particles", type=int, default=1)
    parser.add_argument("--max_particles", type=int, default=5)
    parser.add_argument("--particle_radius", type=int, default=40, help="Particle radius in pixels")
    parser.add_argument("--output_dir", type=str, default="dataset/dummy_dataset")
    parser.add_argument("--max_overlap", type=float, default=0.0, help="Maximum allowed overlap of"
                                                                       "particles. 0 is no overlap 1 is full overlap")
    parser.add_argument("--noise", type=float, default=0.8, help="How much the generated image should be"
                                                                 "blended with random noise, 0 is none 1 means the "
                                                                 "generated image will be fully noisy")
    parser.add_argument("--uniform_noise", type=bool, default=True, help="If set to true the noise added "
                                                                         "to the image will be the same for all color "
                                                                         "channels")

    args = parser.parse_args()
    return args


def create_dummy_dataset(args):
    """
    Creates a dummy dataset and saves it.

    :param args: argparse.ArgumentParser object with the following args:
        image_size: Size of images that are created
        num_images: Number of images to create
        min_particles: Minimum number of particles per image
        max_particles: Maximum number of particles per image
        particle_radius: Size of particles that are drawn
        output_dir: Directory to save the dataset
        max_overlap: Maximum allowed percentage overlap between particles (0.0 to 1.0)
        noise: Level of Gaussian noise to add (0.0 = no noise, 1.0 = fully noisy)
        uniform_noise: Boolean, If set to true the noise added to the image will be the same for all color channels
    """
    # Adjust output_dir to include max_overlap, noise, and uniform_noise
    args.output_dir = (f"{args.output_dir}_max_overlap_{args.max_overlap}_noise_{args.noise}_uniform_noise_"
                       f"{args.uniform_noise}")

    create_folder_if_missing(args.output_dir)
    create_folder_if_missing(os.path.join(args.output_dir, 'data'))

    readme_path = os.path.join(args.output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    for i in tqdm(range(args.num_images), desc="Creating images"):
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, args.image_size)
        ax.set_ylim(0, args.image_size)
        ax.axis('off')

        coordinates = []
        num_dots = random.randint(args.min_particles, args.max_particles)
        for _ in range(num_dots):
            counter = 0
            # Check 
            while True:
                x = np.random.randint(0, args.image_size)
                y = np.random.randint(0, args.image_size)
                new_circle = patches.Circle((x, y), radius=args.particle_radius, color='black')

                # Check overlap with existing particles
                overlap = False
                for coord in coordinates:
                    dist = np.sqrt((x - coord[0])**2 + (y - coord[1])**2)
                    if dist < 2 * args.particle_radius * (1 - args.max_overlap):
                        overlap = True
                        break

                if not overlap:
                    coordinates.append((x, y))
                    ax.add_patch(new_circle)
                    break

                if counter > 100:
                    print(f"Too many attempts to place a particle without overlap. Skipping one patch at image {i}")
                    break

        # Save the image with noise
        image_path = os.path.join(args.output_dir, f'data/micrograph_{i}.png')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # ARGB has 4 channels
        image = image[:, :, 1:]  # Drop the alpha channel to get RGB
        plt.close(fig)

        # Add Gaussian noise
        if args.noise > 0.0:
            # TODO: fix this noise implementation, its weird, use SNR
            # Generate random noise image
            if args.uniform_noise:
                random_noise = np.random.randint(0, 256, (image.shape[0], image.shape[1], 1), dtype=np.uint8)
                random_noise = np.repeat(random_noise, 3, axis=2)  # Make noise uniform across channels
            else:
                random_noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
            
            # Blend the original image with the random noise
            image = (1 - args.noise) * image + args.noise * random_noise
            image = np.clip(image, 0, 255).astype(np.uint8)  # TODO: this is wrong now

        Image.fromarray(image).save(image_path)

        coordinates_path = os.path.join(args.output_dir, f'data/micrograph_{i}_coords.txt')
        with open(coordinates_path, 'w') as f:
            for coord in coordinates:
                f.write(f'{coord[0]},{coord[1]}\n')


class DummyDataset(Dataset):
    def __init__(self, dataset_size, particle_width, particle_height, dataset_path, image_width=224, image_height=224):
        self.dataset_path = dataset_path
        self.particle_width = particle_width
        self.particle_height = particle_height
        self.image_width = image_width
        self.image_height = image_height

        self.micrographs = []
        self.targets = []

        for idx in range(dataset_size):

            # The image (micrograph)
            micrograph_path = os.path.join(self.dataset_path, f'micrograph_{idx}.png')
            if not os.path.isfile(micrograph_path):
                raise Exception(f"The file {micrograph_path} doesn't exist.")

            image = Image.open(micrograph_path).convert("RGB")
            transform = transforms.ToTensor()
            image_tensor = transform(image)

            self.micrographs.append(image_tensor)

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


def get_particle_locations_from_coordinates(coordinates_tl, sub_micrograph_size, particle_width, particle_height, particle_locations, z_slice_size,
                                            orientation="normal"):
    """
    Given coordinates, this function determines the location of all relevant particles in the sub micrograph

    :param coordinates_tl: A tensor with coordinates of original micrograph of the top left most point in the
    sub micrograph. Index 0 is the x value and index 1 is the y value.
    :param sub_micrograph_size: Size of the sub micrographs, usually 224
    :param particle_locations: Pandas DataFrame containing all particle locations of the micrograph needs x,y,z columns
    :param orientation: TODO
    :return: A dictionary with "boxes" key containing a tensor of bounding boxes for particles present in the
        sub micrograph determined by coordinate_tl. The coordinates are scaled to be 0 if theyre on the top left point
        of the sub micrograph
    """
    if orientation == "normal":
        x_min = coordinates_tl[0].item()
        x_max = x_min + sub_micrograph_size
        y_min = coordinates_tl[1].item()
        y_max = y_min + sub_micrograph_size
        z_min = coordinates_tl[2].item()
        z_max = z_min + z_slice_size

        selected_particles = particle_locations[(particle_locations['X'] >= x_min) &
                                                (particle_locations['X'] <= x_max) &
                                                (particle_locations['Y'] >= y_min) &
                                                (particle_locations['Y'] <= y_max) &
                                                (particle_locations['Z'] >= z_min) &
                                                (particle_locations['Z'] <= z_max)]

        # We subtract the minimum coordinates since we want the locations in the sub_micrograph so to speak
        selected_particles.loc[:, 'X'] = (selected_particles['X'] - x_min)
        selected_particles.loc[:, 'Y'] = (selected_particles['Y'] - y_min)

        # Remove the "Z" column
        selected_particles = selected_particles.drop(columns=['Z'])

        # Convert to dictionary with "boxes" key
        boxes = torch.tensor(selected_particles[['X', 'Y']].values, dtype=torch.float32)
        particle_sizes = torch.tensor([particle_width, particle_height]).repeat(len(boxes), 1)
        boxes = torch.cat((boxes, particle_sizes), dim=1)

        return {"boxes": boxes}
    else:
        raise Exception(f'The orientation {orientation} is not a valid orientation')


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


def create_sub_micrographs(micrograph, crop_size, sampling_points, start_z):
    """
    Creates sub micrographs of the given micrograph by sliding a window of size crop_size x crop_size across the image
    from the top left to the bottom right. The total number of sub micrographs will be samping_points * sampling_points

    :param micrograph: A tensor of the micrograph to take sub micrographs from.
    :param crop_size: size of the sliding window
    :param sampling_points: The amount of stops the sliding window takes on one side of the micrograph.
    :param: start_z: Starting z for all submicrographs created for the given micrograph
    :return: A dataframe where the first column is a tensor of sub micrograph and the second a tensor with
    the coordinates of the top left most point of that sub micrograph in the original picture.
    """
    height, width = micrograph.shape

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"

    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    sub_micrographs_list = []
    for i in range(sampling_points):  # horizontal steps
        for j in range(sampling_points):  # vertical steps
            start_x = i * step_size_x
            start_y = j * step_size_y
            end_y = start_y + crop_size
            end_x = start_x + crop_size

            # Ensure we don't go out of bounds
            if end_x <= width and end_y <= height:
                sub_micrograph = micrograph[start_y:end_y, start_x:end_x].unsqueeze(0)
                if micrograph.max() > 0:  # We don't need to normalize if everything is 0
                    sub_micrograph = (sub_micrograph - sub_micrograph.min()) / (sub_micrograph.max() -
                                                                                sub_micrograph.min())
                    if sub_micrograph.max() < 0:  # If its zero after normalization then we don't need it
                        continue
                else:
                    continue
                sub_micrograph = sub_micrograph.repeat(3, 1, 1)
                sub_micrographs_list.append((sub_micrograph, torch.tensor([start_x, start_y, start_z])))

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "top_left_coordinates"])

    return sub_micrographs


class ShrecDataset(Dataset):
    def __init__(self, sampling_points, z_slice_size, particle_width, particle_height, micrograph_size=512,
                 sub_micrograph_size=224, model_number=1, dataset_path='dataset/shrec21_full_dataset/', min_z=140,
                 max_z=360):
        """
        Dataset Loader for Shrec21 Dataset.

        :param sampling_points: Determines number of sub_micrographs, sampling_points^2 = number of sub micrographs
        :param z_slice_size: Size of z slices of the grandmodel
        :param micrograph_size: See shrec dataset
        :param sub_micrograph_size: The size we want our sub micrographs to be
        :param model_number: Model to select for this iteration
        :param dataset_path: Path to dataset
        :param min_z: z slices start from here
        :param max_z: z slices end here
        """

        self.dataset_path = dataset_path
        self.particle_width = particle_width
        self.particle_height = particle_height
        self.model_number = model_number
        self.micrograph_size = micrograph_size  # This is only needed for creating the sub micrographs
        self.z_slice_size = z_slice_size
        self.min_z = min_z
        self.max_z = max_z
        self.sub_micrograph_size = sub_micrograph_size
        self.sampling_points = sampling_points

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        sep=r'\s+', names=columns).drop(columns=['rotation_Z1', 'rotation_X',
                                                                 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, f'model_{self.model_number}/grandmodel.mrc'),
                      permissive=True) as f:
            self.grandmodel = torch.tensor(f.data)  # shape [512, 512, 512]

        self.micrographs = []
        self.sub_micrographs = pd.DataFrame(columns=["sub_micrograph", "top_left_coordinates"])

        for i in range((max_z - min_z) // self.z_slice_size):  # TOOD: theres a small piece at the top that is not being made into a micrograph because of the floor division, make sure this is ok
            start_z = min_z + (i * self.z_slice_size)
            end_z = start_z + z_slice_size
            micrograph = self.grandmodel[start_z:end_z].sum(dim=0)  # We know 0 is correct from testing, 0 is top view
            self.micrographs.append((micrograph, start_z))
            sub_micrographs_df = create_sub_micrographs(micrograph, self.sub_micrograph_size, self.sampling_points,
                                                        start_z=start_z)
            self.sub_micrographs = pd.concat([self.sub_micrographs, sub_micrographs_df], ignore_index=True)

    def __len__(self):
        return len(self.sub_micrographs)

    def __getitem__(self, idx):
        """
        Returns two tensors, one with the sub micrograph and one with the coordinates.
        :param idx: The index to take from
        """
        sub_micrograph_entry = self.sub_micrographs.iloc[idx]
        return (sub_micrograph_entry['sub_micrograph'],
                sub_micrograph_entry['top_left_coordinates'])


if __name__ == "__main__":
    args = get_args()
    create_dummy_dataset(args)
