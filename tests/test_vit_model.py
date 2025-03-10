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
from plotting import save_image_with_bounding_object
from main import get_latent_representation

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'
TEST_RESULTS_FOLDER = 'test_vit_model'
TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8
EXAMPLE_VISUALIZATIONS = 3
DATASET_SIZE = 50


def compare_image_cosine_similarity(image_name_1, image_name_2, vit_model):
    """
    Comparison used in vit model tests, don't use it anywhere else
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img1_path = os.path.join('mocks', 'test_vit_model', image_name_1)
    img2_path = os.path.join('mocks', 'test_vit_model', image_name_2)

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)

    output1 = vit_model(img1_tensor)
    output2 = vit_model(img2_tensor)

    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    similarity_score = cosine_similarity.item()

    print(f"Cosine similarity between images {image_name_1} and {image_name_2}: {similarity_score}")


class VitModelTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.dataset = DummyDataset(dataset_size=DATASET_SIZE, dataset_path=TEST_DATASET_PATH)
        self.vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
        self.vit_model.forward = types.MethodType(get_latent_representation, self.vit_model)
        self.vit_model.eval()

    def test_vit_model_forward_shape(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        output = self.vit_model(dummy_input)
        self.assertEqual(output.shape, (1, 768))

    def test_inputs_to_vit(self):
        compare_image_cosine_similarity("small_particles_dummy_1.png", "small_particles_dummy_2.png",
                                        self.vit_model)
        compare_image_cosine_similarity("cat_particles_dummy_1.png", "cat_particles_dummy_2.png",
                                        self.vit_model)
        compare_image_cosine_similarity("big_particles_dummy_1.png", "big_particles_dummy_2.png",
                                        self.vit_model)


if __name__ == '__main__':
    unittest.main()
