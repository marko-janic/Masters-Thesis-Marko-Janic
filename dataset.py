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


def add_noise_to_micrograph(micrograph: torch.Tensor, noise_db: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the micrograph to achieve the specified SNR (in dB).
    SNR defined here: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    Args:
        micrograph (torch.Tensor): Input tensor of shape (H, W) or (B, C, H, W).
        noise_db (float): Desired SNR in dB (signal-to-noise ratio).

    Returns:
        torch.Tensor: Noisy micrograph tensor with the same shape as input.
    """
    # Flatten to (N, *) for batch processing
    orig_shape = micrograph.shape
    if micrograph.dim() == 2:
        signal = micrograph
    else:
        signal = micrograph.view(-1, *micrograph.shape[-2:])

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
    if micrograph.dim() == 2:
        return noisy_signal
    else:
        return noisy_signal.view(orig_shape)


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
            pass  # TODO: implement this

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
    :param start_z: Starting z for all submicrographs created for the given micrograph
    :return: A dataframe where the first column is a tensor of sub micrograph and the second a tensor with
    the coordinates of the top left most point of that sub micrograph in the original picture.
    """
    height, width = micrograph.shape

    assert sampling_points <= (width - crop_size), "Number of sampling points can't be larger than width - crop_size"

    step_size_x = (width - crop_size) // (sampling_points - 1)
    step_size_y = (height - crop_size) // (sampling_points - 1)

    sub_micrographs_list = []
    if micrograph.max() > micrograph.min():  # We don't want useless images
        for i in range(sampling_points):  # horizontal steps
            for j in range(sampling_points):  # vertical steps
                start_x = i * step_size_x
                start_y = j * step_size_y
                end_y = start_y + crop_size
                end_x = start_x + crop_size

                # Ensure we don't go out of bounds
                if end_x <= width and end_y <= height:
                    sub_micrograph = micrograph[start_y:end_y, start_x:end_x].unsqueeze(0)
                    if sub_micrograph.max() > sub_micrograph.min():  # We don't need to normalize if everything is 0
                        sub_micrograph = (sub_micrograph - sub_micrograph.min()) / (sub_micrograph.max() -
                                                                                    sub_micrograph.min())
                    sub_micrograph = sub_micrograph.repeat(3, 1, 1)

                    if not torch.isnan(sub_micrograph).all():
                        sub_micrographs_list.append((sub_micrograph, torch.tensor([start_x, start_y, start_z])))
                    else:
                        print("One nan tensor found")

    # The reason we did a list first is because of this:
    # https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    sub_micrographs = pd.DataFrame(sub_micrographs_list, columns=["sub_micrograph", "top_left_coordinates"])

    return sub_micrographs


def create_3d_gaussian_volume(particle_locations: pd.DataFrame, particle_width, particle_height, device, amplitude=1.0,
                              volume_shape=(512, 512, 512)):
    """
    Creates a 3D tensor with 3D Gaussians centered at each particle location.
    Only computes each Gaussian in a local window around its center.

    :param particle_locations: DataFrame with columns ['X', 'Y', 'Z'] (must be in voxel coordinates)
    :param volume_shape: Shape of the output tensor (default: 512x512x512)
    :param particle_width: Standard deviation for X axis (in voxels) and Z axis
    :param particle_height: Standard deviation for Y axis (in voxels)
    :param device:
    :param amplitude: Peak value of each Gaussian
    :return: 3D torch.Tensor of shape volume_shape
    """
    volume = torch.zeros(volume_shape, dtype=torch.float32, device=device)

    # Magic numbers for sigmas :3
    sigma_x = particle_width / 3.5  # TODO: maybe add these as arguments?
    sigma_y = particle_height / 3.5
    sigma_z = sigma_x  # Isotropic in Z

    # Define window size (e.g., 3 sigma in each direction), we do this to save computation time for the gaussians
    wx = int(3 * sigma_x)
    wy = int(3 * sigma_y)
    wz = int(3 * sigma_z)

    for _, row in particle_locations.iterrows():
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


def generate_projections(vol,angles):
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
    def __init__(self, sampling_points, z_slice_size, particle_width, particle_height, noise, gaussians_3d,
                 add_noise=False, micrograph_size=512, sub_micrograph_size=224, model_number=1,
                 dataset_path='dataset/shrec21_full_dataset/', min_z=140, max_z=360, device="cpu", use_fbp=False,
                 fbp_min_angle=-torch.pi/3, fbp_max_angle=torch.pi/3, fbp_num_projections=60):
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
        :param add_noise: If noise should be added to the micrographs or not
        :param gaussians_3d: whether to use 3d gaussians or not
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
        self.model_number = model_number
        self.micrograph_size = micrograph_size  # This is only needed for creating the sub micrographs
        self.z_slice_size = z_slice_size
        self.min_z = min_z
        self.max_z = max_z
        self.sub_micrograph_size = sub_micrograph_size
        self.sampling_points = sampling_points
        self.noise = noise
        self.add_noise = add_noise
        self.gaussians_3d = gaussians_3d
        self.device = device
        self.use_fbp = use_fbp
        self.fbp_min_angle = fbp_min_angle
        self.fbp_max_angle = fbp_max_angle
        self.fbp_num_projections = fbp_num_projections

        columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
        self.particle_locations = (
            pd.read_csv(os.path.join(self.dataset_path, f'model_{self.model_number}/particle_locations.txt'),
                        sep=r'\s+', names=columns).drop(columns=['rotation_Z1', 'rotation_X',
                                                                 'rotation_Z2']))

        with mrc.open(os.path.join(self.dataset_path, f'model_{self.model_number}/grandmodel.mrc'),
                      permissive=True) as f:
            self.grandmodel = torch.tensor(f.data)  # shape [512, 512, 512]

        if self.gaussians_3d:
            self.heatmaps_volume = create_3d_gaussian_volume(particle_locations=self.particle_locations,
                                                             particle_width=self.particle_width,
                                                             particle_height=self.particle_height, amplitude=1.0,
                                                             device=self.device)

        if self.use_fbp:
            angles = np.linspace(self.fbp_min_angle, self.fbp_max_angle, fbp_num_projections)
            projections = generate_projections(self.grandmodel, angles)
            self.grandmodel_fbp = reconstruct_fbp_volume(projections, angles, self.grandmodel.shape[0])

        self.micrographs = []
        self.sub_micrographs = pd.DataFrame(columns=["sub_micrograph", "top_left_coordinates"])

        if self.gaussians_3d:
            self.heatmaps = []
            self.sub_heatmaps = pd.DataFrame(columns=["sub_micrograph", "top_left_coordinates"])

        for i in range((max_z - min_z) // self.z_slice_size):
            start_z = min_z + (i * self.z_slice_size)
            end_z = start_z + z_slice_size

            if not self.use_fbp:
                # We know 0 is correct from testing, 0 is top view so to speak
                micrograph = self.grandmodel[start_z:end_z].sum(dim=0)
            else:
                micrograph = self.grandmodel_fbp[start_z:end_z].sum(dim=0)

            # TODO: Change this, we don't add noise to the reconstrution but rather to the projections themselves
            if add_noise:
                micrograph = add_noise_to_micrograph(micrograph=micrograph, noise_db=self.noise)

            self.micrographs.append((micrograph, start_z, end_z))
            sub_micrographs_df = create_sub_micrographs(micrograph, self.sub_micrograph_size, self.sampling_points,
                                                        start_z=start_z)
            self.sub_micrographs = pd.concat([self.sub_micrographs, sub_micrographs_df], ignore_index=True)

            if self.gaussians_3d:
                # Here we max not sum since we are trying to represent probabilities for the targets
                heatmap = self.heatmaps_volume[start_z:end_z].max(dim=0).values
                self.heatmaps.append((heatmap, start_z, end_z))

                sub_heatmaps_df = create_sub_micrographs(heatmap, self.sub_micrograph_size, self.sampling_points,
                                                         start_z=start_z)
                # Resize each heatmap in the DataFrame to (112, 112), we do this weird squeezing because F.interpolate
                # needs batched input. We take [0:1] because create_sub_micrographs automatically adds three channels
                sub_heatmaps_df["sub_micrograph"] = sub_heatmaps_df["sub_micrograph"].apply(
                    lambda x: F.interpolate(x[0:1].unsqueeze(0), size=(112, 112), mode='bilinear',
                                            align_corners=False).squeeze(0)
                )
                self.sub_heatmaps = pd.concat([self.sub_heatmaps, sub_heatmaps_df], ignore_index=True)

    def get_target_heatmaps_from_3d_gaussians(self, tl_coordinates, batch_size):
        """
        Given a batch of top-left coordinates, return the corresponding heatmaps as a tensor of shape
        [batch_size, 1, heatmap_size, heatmap_size].
        :param tl_coordinates: Tensor of shape [batch_size, 3] (x, y, z)
        :param batch_size: Number of samples in the batch
        :return: torch.Tensor of shape [batch_size, 1, 3, heatmap_size, heatmap_size]
        """
        # Ensure input is a tensor
        if not torch.is_tensor(tl_coordinates):
            tl_coordinates = torch.tensor(tl_coordinates)

        # Prepare output list
        heatmaps = []
        for i in range(batch_size):
            coord = tl_coordinates[i]
            # Find the row in sub_heatmaps with matching top_left_coordinates
            matches = self.sub_heatmaps['top_left_coordinates'].apply(
                lambda x: torch.equal(x, coord)
            )
            idx = matches.idxmax() if matches.any() else None
            if idx is not None:
                heatmap = self.sub_heatmaps.iloc[idx]['sub_micrograph']
                heatmaps.append(heatmap)
            else:
                raise Exception("The top left coordinates don't have equivalent target_heatmaps from the 3d gaussian"
                                "volume. This should not happen :3, have fun debugging.")

        return torch.stack(heatmaps, dim=0)

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
    dummy_args = get_args()
    create_dummy_dataset(dummy_args)
