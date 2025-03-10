import unittest
import os
import torch
import types

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.patches as patches

from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.models import vit_b_16

# Local imports
from dataset import DummyDataset
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import save_image_with_bounding_object, compare_predictions_with_ground_truth
from model import ParticlePicker
from main import get_latent_representation
from train import prepare_outputs_for_loss

TEST_DATASET_PATH = '../dataset/dummy_dataset_old/data'
TEST_RESULTS_FOLDER = 'test_plotting'
MOCKS_FOLDER = 'mocks'
EXPERIMENT_FOLDER = os.path.join(MOCKS_FOLDER, "experiment_dummy_dataset")
CHECKPOINT_PATH = os.path.join(EXPERIMENT_FOLDER, "checkpoints/checkpoint_untrained.pth")

TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8
EXAMPLE_VISUALIZATIONS = 2
DATASET_SIZE = 50

LATENT_DIMENSION = 768
NUM_PARTICLES = 50


class PlottingTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)

        self.dataset = DummyDataset(dataset_size=DATASET_SIZE, dataset_path=TEST_DATASET_PATH)

        self.model = ParticlePicker(LATENT_DIMENSION, NUM_PARTICLES, self.dataset.image_width, self.dataset.image_height)
        self.model.eval()
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH))

        self.vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
        self.vit_model.eval()
        self.vit_model.forward = types.MethodType(get_latent_representation, self.vit_model)

    def test_save_image_with_bounding_object(self):
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

    def test_compare_predictions_with_ground_truth(self):
        for i in range(EXAMPLE_VISUALIZATIONS):
            micrograph, target_index = self.dataset.__getitem__(i)
            coordinates = self.dataset.targets[target_index]['boxes'][:, :4]
            coordinates = transform_coords_to_pixel_coords(image_width=self.dataset.image_width,
                                                           image_height=self.dataset.image_height, coords=coordinates)

            latent_micrograph = self.vit_model(micrograph.unsqueeze(0))
            predictions = self.model(latent_micrograph)
            outputs = prepare_outputs_for_loss(predictions)

            output_coordinates = transform_coords_to_pixel_coords(image_width=self.dataset.image_width,
                                                                  image_height=self.dataset.image_height,
                                                                  coords=outputs["pred_boxes"][0])
            output_coordinates = output_coordinates.detach().numpy()

            compare_predictions_with_ground_truth(micrograph, coordinates, output_coordinates, "circle",
                                                  {"box_width": self.dataset.particle_width,
                                                   "box_height": self.dataset.particle_height, "circle_radius": 4},
                                                  TEST_RESULTS_FOLDER, f"comparison_{i}.png")


if __name__ == '__main__':
    unittest.main()
