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
import torch.nn.functional as F
import tomosipo as ts

from ts_algorithms import fbp
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

# Local imports
from utils import create_folder_if_missing


def add_noise_to_projections(projections: torch.Tensor, noise_db: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the micrograph to achieve the specified SNR (in dB).
    SNR defined here: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    Args:
        projections (torch.Tensor): Input tensor of shape (num_projections, H, W)
        noise_db (float): Desired SNR in dB (signal-to-noise ratio)

    Returns:
        torch.Tensor: Noisy micrograph tensor with the same shape as input.
    """
    signal = projections

    # Compute signal power
    signal_power = signal.pow(2).mean(dim=(-2, -1), keepdim=True)

    # Compute noise power for desired SNR
    snr_linear = 10 ** (noise_db / 10)
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = torch.randn_like(signal) * noise_power.sqrt()

    # Add noise
    noisy_signal = signal + noise

    # Reshape back to original
    return noisy_signal


def get_particle_locations_from_coordinates(coordinates_tl, sub_micrograph_size, particle_width, particle_height,
                                            shrec_specific_particle, particle_depth, particle_locations, z_slice_size,
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
    if orientation in ["normal", "vertical_flip", "horizontal_flip", "transposed"]:
        x_min = coordinates_tl[0].item() - (particle_width * 0.5)
        x_max = x_min + sub_micrograph_size + (particle_width * 0.5)

        y_min = coordinates_tl[1].item() - (particle_height * 0.5)
        y_max = y_min + sub_micrograph_size + (particle_height * 0.5)

        # This is just to help with visualizing actual particles during training, only for debugging
        z_min = coordinates_tl[2].item() - (particle_depth * 0.5)
        z_max = z_min + z_slice_size + (particle_depth * 0.5)

        # Shrec dataset is bugged, so we exclude particle 4V94, see important note here: https://www.shrec.net/cryo-et/
        selected_particles = particle_locations[(particle_locations['X'] >= x_min) &
                                                (particle_locations['X'] <= x_max) &
                                                (particle_locations['Y'] >= y_min) &
                                                (particle_locations['Y'] <= y_max) &
                                                (particle_locations['Z'] >= z_min) &
                                                (particle_locations['Z'] <= z_max) &
                                                (particle_locations['class'] != "4V94") &
                                                (particle_locations['class'] != "vesicle")].copy()  # Copy to avoid SettingWithCopyWarning

        # Conditional filter for shrec_specific_particle
        if shrec_specific_particle is not None and shrec_specific_particle != "":
            selected_particles = selected_particles[selected_particles['class'] == shrec_specific_particle]

        # We subtract the minimum coordinates since we want the locations in the sub_micrograph so to speak
        selected_particles.loc[:, 'X'] = (selected_particles['X'] - x_min)
        selected_particles.loc[:, 'Y'] = (selected_particles['Y'] - y_min)

        # Remove the "Z" column
        selected_particles = selected_particles.drop(columns=['Z'])

        if orientation == "normal":
            pass
        elif orientation == "vertical_flip":
            selected_particles['Y'] = sub_micrograph_size - selected_particles['Y']
        elif orientation == "horizontal_flip":
            selected_particles['X'] = sub_micrograph_size - selected_particles['X']
        elif orientation == "transposed":
            # Transpose means we swap X and Y
            x_df = selected_particles['X'].copy()
            y_df = selected_particles['Y'].copy()
            selected_particles['X'] = y_df
            selected_particles['Y'] = x_df

        # Convert to dictionary with "boxes" key
        boxes = torch.tensor(selected_particles[['X', 'Y']].values, dtype=torch.float32)
        particle_sizes = torch.tensor([particle_width, particle_height]).repeat(len(boxes), 1)
        boxes = torch.cat((boxes, particle_sizes), dim=1)

        return boxes
    else:
        raise Exception(f'The orientation {orientation} is not a valid orientation')


def create_sub_micrographs(micrograph, heatmap, crop_size, sampling_points, start_z, model_number, sub_heatmap_size,
                           random_sub_micrographs):
    """
    Creates sub micrographs of the given micrograph by sliding a window of size crop_size x crop_size across the image
    from the top left to the bottom right. The total number of sub micrographs will be samping_points * sampling_points

    :param micrograph: A tensor of the micrograph to take sub micrographs from.
    :param crop_size: size of the sliding window
    :param sampling_points: The amount of stops the sliding window takes on one side of the micrograph.
    :param start_z: Starting z for all submicrographs created for the given micrograph
    :return: A dataframe where the first column is a tensor of sub micrograph and the second a tensor with
    the coordinates of the top left most point of that sub micrograph in the original picture.
    """
    height, width = micrograph.shape

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"
    assert micrograph.shape == heatmap.shape, "Micrograph and heatmap must have the same shape"

    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    sub_micrographs_list = []
    for i in range(sampling_points):  # horizontal steps
        for j in range(sampling_points):  # vertical steps
            if not random_sub_micrographs:
                start_x = i * step_size_x
                start_y = j * step_size_y
                end_y = start_y + crop_size
                end_x = start_x + crop_size
            else:
                start_x = torch.randint(0, width - crop_size, (1,)).item()
                start_y = torch.randint(0, height - crop_size, (1,)).item()
                end_y = start_y + crop_size
                end_x = start_x + crop_size

            # Ensure we don't go out of bounds
            if end_x <= width and end_y <= height:
                sub_micrograph = micrograph[start_y:end_y, start_x:end_x].unsqueeze(0)
                if sub_micrograph.max() > sub_micrograph.min():  # We don't need to normalize if everything is 0
                    sub_micrograph = (sub_micrograph - sub_micrograph.min()) / (sub_micrograph.max() -
                                                                                sub_micrograph.min())
                sub_micrograph = sub_micrograph.repeat(3, 1, 1)

                sub_heatmap = heatmap[start_y:end_y, start_x:end_x].unsqueeze(0)
                # Resize each heatmap in the DataFrame to (112, 112) since the model outputs heatmaps of that size
                # We do this weird squeezing because F.interpolate needs batched input
                # We take [0:1] because we want to preserve the channel dimension
                sub_heatmap = F.interpolate(sub_heatmap[0:1].unsqueeze(0), size=(sub_heatmap_size, sub_heatmap_size),
                                            mode='bilinear', align_corners=False).squeeze(0)

                if not torch.isnan(sub_micrograph).all():
                    # Normal
                    sub_micrographs_list.append((
                        sub_micrograph,
                        sub_heatmap,
                        torch.tensor([start_x, start_y, start_z]), "normal", model_number))

                    # Horizontal flip
                    sub_micrographs_list.append((
                        torch.flip(sub_micrograph, dims=[2]),
                        torch.flip(sub_heatmap, dims=[2]),
                        torch.tensor([start_x, start_y, start_z]), "horizontal_flip", model_number))

                    # Vertical flip
                    sub_micrographs_list.append((
                        torch.flip(sub_micrograph, dims=[1]),
                        torch.flip(sub_heatmap, dims=[1]),
                        torch.tensor([start_x, start_y, start_z]), "vertical_flip", model_number))

                    # Transposed version
                    sub_micrographs_list.append((
                        sub_micrograph.permute(0, 2, 1),
                        sub_heatmap.permute(0, 2, 1),
                        torch.tensor([start_x, start_y, start_z]), "transposed", model_number))

                else:
                    raise Exception("One nan tensor found")
            else:
                raise Exception("Ending x or y is not within bounds of the micrograph")

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "heatmap", "top_left_coordinates",
                                                                  "orientation", "model_number"])
    return sub_micrographs


def create_3d_gaussian_volume(particle_locations: pd.DataFrame, particle_width, particle_height, particle_depth, device,
                              amplitude=1.0, volume_shape=(512, 512, 512), shrec_specific_particle=None):
    """
    Creates a 3D tensor with 3D Gaussians centered at each particle location.
    Only computes each Gaussian in a local window around its center.

    :param particle_locations: DataFrame with columns ['X', 'Y', 'Z'] (must be in voxel coordinates)
    :param volume_shape: Shape of the output tensor (default: 512x512x512)
    :param particle_width: Standard deviation for X axis (in voxels) and Z axis
    :param particle_height: Standard deviation for Y axis (in voxels)
    :param particle_depth: Determines STD for z axis
    :param device:
    :param amplitude: Peak value of each Gaussian
    :param shrec_specific_particle Only use this particle for the heatmaps
    :return: 3D torch.Tensor of shape volume_shape
    """
    volume = torch.zeros(volume_shape, dtype=torch.float32, device=device)

    for _, row in particle_locations.iterrows():
        class_name = str(row['class'])
        # Shrec dataset is bugged, so we exclude particle 4V94, see important note here: https://www.shrec.net/cryo-et/
        # Vesicle also seems to just be invisible?
        if class_name == "4V94" or class_name == "vesicle":
            continue

        if class_name == "5MRC":
            sigma_x = (particle_width + 8) / 3
            sigma_y = (particle_height + 8) / 3
            sigma_z = (particle_depth + 5) / 3
            wx = int(3 * sigma_x)
            wy = int(3 * sigma_y)
            wz = int(3 * sigma_z)
        elif class_name == "4CR2":
            sigma_x = (particle_width + 2) / 3
            sigma_y = (particle_height + 2) / 3
            sigma_z = (particle_depth + 3) / 3
            wx = int(3 * sigma_x)
            wy = int(3 * sigma_y)
            wz = int(3 * sigma_z)
        else:
            sigma_x = particle_width / 3
            sigma_y = particle_height / 3
            sigma_z = particle_depth / 3
            # Define window size (e.g., 3 sigma in each direction), we do this to save computation time for
            # the gaussians
            wx = int(3 * sigma_x)
            wy = int(3 * sigma_y)
            wz = int(3 * sigma_z)

        if shrec_specific_particle is None or shrec_specific_particle == "":
            pass
        else:
            if class_name != shrec_specific_particle:  # Only make the particle for the specified particle
                continue

        cx, cy, cz = int(row['X']), int(row['Y']), int(row['Z'])

        # Define bounds, making sure they are within the volume
        x_min, x_max = max(cx - wx, 0), min(cx + wx + 1, volume_shape[2])
        y_min, y_max = max(cy - wy, 0), min(cy + wy + 1, volume_shape[1])
        z_min, z_max = max(cz - wz, 0), min(cz + wz + 1, volume_shape[0])

        # Create local grids
        z = torch.arange(z_min, z_max, device=device).view(-1, 1, 1)
        y = torch.arange(y_min, y_max, device=device).view(1, -1, 1)
        x = torch.arange(x_min, x_max, device=device).view(1, 1, -1)

        # Compute local Gaussian
        gaussian = amplitude * torch.exp(
            -(((x - cx) ** 2) / (2 * sigma_x ** 2) +
              ((y - cy) ** 2) / (2 * sigma_y ** 2) +
              ((z - cz) ** 2) / (2 * sigma_z ** 2))
        )

        # Update only the local region
        volume[z_min:z_max, y_min:y_max, x_min:x_max] = torch.max(
            volume[z_min:z_max, y_min:y_max, x_min:x_max], gaussian
        )

    return volume


def generate_projections(vol, angles):
    """
    Given a volume and a set of angles this will simulate projections at the specified angles.
    :param vol: torch.Tensor[D, H, W]
    :param angles:
    :return:
    """
    # note angles in radians
    n1, n2, n3 = vol.shape
    pg = ts.parallel(angles=angles, shape=(n1, n2),)
    vg = ts.volume(shape=(n1, n3, n2))  # Reordering so that this is samle as ODL
    A = ts.operator(vg, pg)
    projection = A(vol.permute(0, 2, 1))
    return projection.permute(1, 0, 2)


def reconstruct_fbp_volume(projections, angles, n3):
    # Define the forward operator
    # angels in radians
    n1 = projections.shape[1]
    n2 = projections.shape[2]
    pg = ts.parallel(angles=angles, shape=(n1, n2),)
    vg = ts.volume(shape=(n1, n3, n2))  # Reordering so that this is samle as ODL
    A = ts.operator(vg, pg)
    V_FBP = fbp(A, projections.permute(1, 0, 2)).permute(0, 2, 1)
    return V_FBP


class ShrecDataset(Dataset):
    def __init__(self, sampling_points, z_slice_size, particle_width, particle_height, particle_depth, noise,
                 add_noise, heatmap_size, micrograph_size=512, sub_micrograph_size=224, model_number=None,
                 dataset_path='dataset/shrec21_full_dataset/', min_z=140, max_z=360, device="cpu", use_fbp=False,
                 fbp_min_angle=-torch.pi/3, fbp_max_angle=torch.pi/3, fbp_num_projections=60,
                 shrec_specific_particle=None, random_sub_micrographs=False, use_shrec_reconstruction=False):
        """
        Dataset Loader for Shrec21 Dataset.

        :param sampling_points: Determines number of sub_micrographs, sampling_points^2 = number of sub micrographs
        :param z_slice_size: Size of z slices of the grandmodel
        :param micrograph_size: See shrec dataset
        :param sub_micrograph_size: The size we want our sub micrographs to be
        :param model_number: Models to select for this iteration. Needs to be specified as list
        :param dataset_path: Path to dataset
        :param min_z: z slices start from here
        :param max_z: z slices end here
        :param add_noise: If noise should be added to the micrographs or not
        :param noise: The level of noise to add to the individual micrographs
        :param use_fbp: To use a simulated fbp reconstruction of the grandmodel
        :param:fbp_num_projections: Num of measurements to simulate for reconstructing the grandmodel using fbp
        :param fbp_min_angle
        :param fbp_max_angle
        :param device
        """

        self.dataset_path = dataset_path
        self.particle_width = particle_width
        self.particle_height = particle_height
        self.particle_depth = particle_depth
        if model_number is None:  # We do this to avoid mutable default arguments
            self.model_number = [1]
        else:
            self.model_number = model_number
        self.micrograph_size = micrograph_size  # This is only needed for creating the sub micrographs
        self.z_slice_size = z_slice_size
        self.min_z = min_z
        self.max_z = max_z
        self.sub_micrograph_size = sub_micrograph_size
        self.sampling_points = sampling_points
        self.noise = noise
        self.add_noise = add_noise
        self.device = device
        self.use_fbp = use_fbp
        self.fbp_min_angle = fbp_min_angle
        self.fbp_max_angle = fbp_max_angle
        self.fbp_num_projections = fbp_num_projections
        self.shrec_specific_particle = shrec_specific_particle
        self.heatmap_size = heatmap_size
        self.random_sub_micrographs = random_sub_micrographs

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = {}
        for model_num in self.model_number:
            df = pd.read_csv(os.path.join(self.dataset_path, f'model_{model_num}/particle_locations.txt'),
                             sep=r'\s+', names=columns).drop(columns=['rotation_Z1', 'rotation_X', 'rotation_Z2'])
            self.particle_locations[model_num] = df

        self.grandmodel = {}
        for model_num in self.model_number:
            with mrc.open(os.path.join(self.dataset_path, f'model_{model_num}/grandmodel.mrc'),
                          permissive=True) as f:
                self.grandmodel[model_num] = torch.tensor(f.data, dtype=torch.float32)  # shape [512, 512, 512]

        self.heatmaps_volume = {}
        for model_num in self.model_number:
            self.heatmaps_volume[model_num] = create_3d_gaussian_volume(
                particle_locations=self.particle_locations[model_num], particle_width=self.particle_width,
                particle_height=self.particle_height, particle_depth=self.particle_depth, amplitude=1.0,
                device=self.device, shrec_specific_particle=self.shrec_specific_particle)

        if self.use_fbp:
            self.grandmodel_fbp = {}
            if not use_shrec_reconstruction:
                angles = np.linspace(self.fbp_min_angle, self.fbp_max_angle, fbp_num_projections)
                for model_num in self.model_number:
                    projections = generate_projections(self.grandmodel[model_num].permute(2, 1, 0), angles)
                    # We add noise to the projections and then reconstruct to simulate how it would be
                    if self.add_noise:
                        projections = add_noise_to_projections(projections, noise_db=self.noise)
                    self.grandmodel_fbp[model_num] = reconstruct_fbp_volume(
                        projections, angles, self.grandmodel[model_num].shape[0]).permute(2, 1, 0)
            else:
                for model_num in self.model_number:
                    with mrc.open(os.path.join(self.dataset_path, f'model_{model_num}/reconstruction.mrc'),
                                  permissive=True) as f:
                        # shape [512, 512, 512]
                        self.grandmodel_fbp[model_num] = torch.tensor(f.data, dtype=torch.float32)

        self.micrographs = {}
        self.heatmaps = {}
        self.sub_micrographs = pd.DataFrame(columns=["sub_micrograph", "heatmap", "top_left_coordinates", "orientation",
                                                     "model_number"])

        for model_num in self.model_number:
            self.micrographs[model_num] = []
            self.heatmaps[model_num] = []

            for i in tqdm(range((max_z - min_z) // self.z_slice_size), desc=f"Loading Dataset for volume {model_num}"):
                start_z = min_z + (i * self.z_slice_size)
                start_z = start_z if start_z >= 0 else 0
                # We do 2* because we want the window to be above and below the current z so to speak
                end_z = start_z + z_slice_size
                end_z = end_z if end_z <= 512 else 512  # TODO: don't hardcode this 512 and 511

                # Here we get the z slices of the grandmodel volume and the heatmap volume
                if not self.use_fbp:
                    # We know 0 is correct from testing, 0 is top view so to speak
                    micrograph = self.grandmodel[model_num][start_z:end_z].sum(dim=0)
                else:
                    micrograph = self.grandmodel_fbp[model_num][start_z:end_z].sum(dim=0)
                self.micrographs[model_num].append((micrograph, start_z, end_z))
                # Here we max not sum since we are trying to represent probabilities for the targets
                heatmap = self.heatmaps_volume[model_num][start_z:end_z].max(dim=0).values
                self.heatmaps[model_num].append((heatmap, start_z, end_z))

                # Here we create the sub micrographs and their corresponding heatmaps by using a sliding window across
                # the z slice heatmap and micrograph
                sub_micrographs_df = create_sub_micrographs(
                    micrograph=micrograph, heatmap=heatmap, crop_size=self.sub_micrograph_size,
                    sampling_points=self.sampling_points, start_z=start_z, model_number=model_num,
                    sub_heatmap_size=self.heatmap_size, random_sub_micrographs=self.random_sub_micrographs)

                self.sub_micrographs = pd.concat([self.sub_micrographs, sub_micrographs_df], ignore_index=True)

    def get_particle_locations_of_models(self):
        target_coordinates_dict = {}

        for model_num in self.model_number:
            df = self.particle_locations[model_num]

            # Filter out "vesicle" and "4V94"
            if self.shrec_specific_particle is None or self.shrec_specific_particle == "":
                filtered = df[(df['class'] != "vesicle") & (df['class'] != "4V94")]
            else:
                filtered = df[(df['class'] != "vesicle") & (df['class'] != "4V94") &
                              (df['class'] == self.shrec_specific_particle)]

            target_coordinates_dict[model_num] = filtered

        return target_coordinates_dict

    def update_sub_micrographs(self):
        """
        Recomputes all sub micrographs, only use it if random_sub_micrographs is True
        :return:
        """
        if self.random_sub_micrographs:
            self.sub_micrographs = pd.DataFrame(columns=["sub_micrograph", "heatmap", "top_left_coordinates",
                                                         "orientation", "model_number"])
            for model_num in tqdm(self.model_number, desc="Recomputing sub micrographs"):
                for i in range(len(self.micrographs[model_num])):
                    start_z = self.micrographs[model_num][i][1]
                    micrograph = self.micrographs[model_num][i][0]
                    heatmap = self.heatmaps[model_num][i][0]

                    sub_micrographs_df = create_sub_micrographs(
                        micrograph=micrograph, heatmap=heatmap, crop_size=self.sub_micrograph_size,
                        sampling_points=self.sampling_points, start_z=start_z, model_number=model_num,
                        sub_heatmap_size=self.heatmap_size, random_sub_micrographs=self.random_sub_micrographs)

                    self.sub_micrographs = pd.concat([self.sub_micrographs, sub_micrographs_df], ignore_index=True)
        else:
            raise Exception("You are calling the update_sub_micrographs function even though random_sub_micrographs"
                            "is set to False. This doesn't do anything.")

    def __len__(self):
        return len(self.sub_micrographs)

    def __getitem__(self, idx):
        """
        Returns two tensors, one with the sub micrograph and one with the coordinates.
        :param idx: The index to take from
        """
        sub_micrograph_entry = self.sub_micrographs.iloc[idx]

        micrograph = sub_micrograph_entry['sub_micrograph']
        heatmap = sub_micrograph_entry['heatmap']
        coordinates_tl = sub_micrograph_entry['top_left_coordinates']
        orientation = sub_micrograph_entry['orientation']
        model_number = sub_micrograph_entry['model_number']
        targets = get_particle_locations_from_coordinates(coordinates_tl=coordinates_tl, orientation=orientation,
                                                          particle_depth=self.particle_depth,
                                                          particle_width=self.particle_width,
                                                          particle_height=self.particle_height,
                                                          sub_micrograph_size=self.sub_micrograph_size,
                                                          particle_locations=self.particle_locations[model_number],
                                                          z_slice_size=self.z_slice_size,
                                                          shrec_specific_particle=self.shrec_specific_particle)
        targets[:, 1] = self.sub_micrograph_size - targets[:, 1]
        # We normalize it between 0 and 1
        targets /= self.sub_micrograph_size

        return micrograph, heatmap, targets, (coordinates_tl, orientation, model_number)
