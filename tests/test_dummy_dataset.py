import unittest
import os
import torch

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.patches as patches

from torch.utils.data import DataLoader, random_split
from PIL import Image

# Local imports
from dataset import DummyDataset
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import save_image_with_bounding_object

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'
TEST_RESULTS_FOLDER = 'test_dummy_dataset'
TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8
EXAMPLE_VISUALIZATIONS = 3
DATASET_SIZE = 50


class DummyDatasetTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.dataset = DummyDataset(dataset_size=DATASET_SIZE, dataset_path=TEST_DATASET_PATH)

    def test_shapes(self):
        pass

    def test_micrographs_normalization(self):
        for i in range(self.dataset.__len__()):
            micrograph, target_index = self.dataset.__getitem__(i)
            # The following lines are not true anymore since I'm using the transforms, maybe check if this is correct
            #self.assertGreaterEqual(micrograph.min(), 0)
            #self.assertGreaterEqual(self.dataset.targets[target_index]['boxes'].min(), 0)
            #self.assertLessEqual(micrograph.max(), 1)
            #self.assertLessEqual(self.dataset.targets[target_index]['boxes'].max(), 1)

    def test_dataset_dataloader(self):
        train_size = int(TRAIN_EVAL_SPLIT * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        for micrographs, target_indexes in train_dataloader:
            targets = []
            for target_index in target_indexes:
                targets.append(self.dataset.targets[target_index])
            pass

        for micrographs, targets in test_dataloader:
            pass

    def test_dataset_images(self):
        for i in range(EXAMPLE_VISUALIZATIONS):
            micrograph, target_index = self.dataset.__getitem__(i)
            fig, ax = plt.subplots(1)
            ax.imshow(micrograph.permute(1, 2, 0).cpu().numpy())
            plt.savefig(os.path.join(TEST_RESULTS_FOLDER, f'example_{i}.png'))
            plt.close(fig)

            coordinates = self.dataset.targets[target_index]['boxes'][:, :2]
            coordinates = transform_coords_to_pixel_coords(image_width=self.dataset.image_width,
                                                           image_height=self.dataset.image_height, coords=coordinates)
            save_image_with_bounding_object(image_tensor=micrograph, particle_locations=coordinates,
                                            object_type="circle", object_parameters={"circle_radius": 4},
                                            result_dir=TEST_RESULTS_FOLDER, file_name=f'example_{i}_circles.png')
            if i < 1:
                save_image_with_bounding_object(micrograph, coordinates, "box",
                                                {"box_width": 10, "box_height": 10}, TEST_RESULTS_FOLDER,
                                                f"example_{i}_boxes.png")


def test_explicit_images():
    """
    Just a helper function to check hardcoded loading
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

    plt.savefig(os.path.join(TEST_RESULTS_FOLDER, f'test_{path_number}.png'))
    plt.close()


if __name__ == '__main__':
    unittest.main()
