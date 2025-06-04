import unittest
import os
import torch
import argparse

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.patches as patches

from torch.utils.data import DataLoader, random_split
from PIL import Image

# Local imports
from dataset import DummyDataset
from utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import save_image_with_bounding_object


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default='test_dummy_dataset')
    parser.add_argument("--dataset_path", type=str, default='../dataset/dummy_dataset_max_overlap_0.0_noise_0.0/data/')
    parser.add_argument("--train_eval_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--example_visualizations", type=int, default=3)
    parser.add_argument("--dataset_size", type=int, default=50)

    args = parser.parse_args()

    return args


class DummyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()
        create_folder_if_missing(self.args.result_dir)
        self.dataset = DummyDataset(dataset_size=self.args.dataset_size, dataset_path=self.args.dataset_path,
                                    particle_width=40, particle_height=40)

    def test_dataset_dataloader(self):
        train_size = int(self.args.train_eval_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        for micrographs, target_indexes in train_dataloader:
            targets = []
            for target_index in target_indexes:
                targets.append(self.dataset.targets[target_index])
            pass

        for micrographs, targets in test_dataloader:
            pass

    def test_dataset_images(self):
        for i in range(self.args.example_visualizations):
            micrograph, target_index = self.dataset.__getitem__(i)
            fig, ax = plt.subplots(1)
            ax.imshow(micrograph.permute(1, 2, 0).cpu().numpy())
            plt.savefig(os.path.join(self.args.result_dir, f'example_{i}.png'))
            plt.close(fig)

            coordinates = self.dataset.targets[target_index]['boxes'].unsqueeze(0)
            coordinates = transform_coords_to_pixel_coords(image_width=self.dataset.image_width,
                                                           image_height=self.dataset.image_height, coords=coordinates)
            save_image_with_bounding_object(image_tensor=micrograph, particle_locations=coordinates[0],
                                            object_type="circle", object_parameters={"circle_radius": 40},
                                            result_dir=self.args.result_dir, file_name=f'example_{i}_circles.png')
            save_image_with_bounding_object(micrograph, coordinates[0], "box",
                                            {"box_width": 80, "box_height": 80}, self.args.result_dir,
                                            f"example_{i}_boxes.png")


def test_explicit_images(args):
    """
    Just a helper function to check hardcoded loading
    args needs:
    - result_dir
    """
    path_number = 2
    transform = transforms.ToTensor()  # This rescales it to [0, 1]
    micrograph_path = os.path.join(f'../dataset/dummy_dataset_old/data/micrograph_{path_number}.png')
    micrograph = transform(Image.open(micrograph_path))[:3, :, :]

    coordinates_path = f'../dataset/dummy_dataset_old/data/micrograph_{path_number}_coords.txt'
    coordinates = pd.read_csv(coordinates_path, sep=',', header=None, names=['X', 'Y'])
    coordinates = torch.tensor(coordinates.values, dtype=torch.float32)
    particle_sizes = torch.tensor([2, 2], dtype=torch.float32).repeat(
        len(coordinates), 1)
    bboxes = torch.cat((coordinates, particle_sizes), dim=1) / torch.Tensor(
        [224, 224, 224, 224])
    classes = torch.zeros(len(bboxes))  # Zeros since we have one class (particle)

    fig, ax = plt.subplots(1)
    ax.imshow(micrograph.permute(1, 2, 0))
    for coords in coordinates:
        x = coords[0]
        y = 224 - coords[1]

        circle = patches.Circle((x, y), 4, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(circle)

    plt.savefig(os.path.join(args.result_dir, f'test_{path_number}.png'))
    plt.close()


if __name__ == '__main__':
    unittest.main()
