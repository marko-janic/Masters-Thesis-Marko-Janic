import unittest
import os
import torch
import types
import datetime
import json

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from torchvision.models import vit_b_16
from tqdm import tqdm

# Local imports
from dataset import DummyDataset
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_images
from vit_model import get_latent_representation
from train import prepare_dataloaders

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'
TEST_RESULTS_FOLDER = 'test_vit_model'
MODEL_RESULTS_FOLDER = os.path.join(TEST_RESULTS_FOLDER, f'experiments/experiment_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8
EVAL_EXAMPLES = 8
EXAMPLE_VISUALIZATIONS = 3
DATASET_SIZE = 500
EPOCHS = 100
CHECKPOINT_INTERVAL = 1
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


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


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 3, 224, 224)  # Reshape to image dimensions
        return x


class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.fc = nn.Linear(768, 7 * 7 * 256)
        self.relu = nn.ReLU(inplace=True)
        self.reshape = lambda x: x.view(-1, 256, 7, 7)

        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.reshape(x)

        x = self.upconv1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.upconv2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.upconv3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.upconv4(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = self.upconv5(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x


class VitModelTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.dataset = DummyDataset(dataset_size=DATASET_SIZE, dataset_path=TEST_DATASET_PATH)
        self.vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
        self.vit_model.forward = types.MethodType(get_latent_representation, self.vit_model)

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

    def test_decoder(self):
        train_dataloader, test_dataloader = prepare_dataloaders(self.dataset, TRAIN_EVAL_SPLIT, BATCH_SIZE)

        decoder_model = UNetDecoder().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(decoder_model.parameters(), lr=LEARNING_RATE)

        create_folder_if_missing(os.path.join(MODEL_RESULTS_FOLDER, "checkpoints"))

        # Save training parameters to a file
        params = {
            "TRAIN_EVAL_SPLIT": TRAIN_EVAL_SPLIT,
            "BATCH_SIZE": BATCH_SIZE,
            "EVAL_EXAMPLES": EVAL_EXAMPLES,
            "DATASET_SIZE": DATASET_SIZE,
            "EPOCHS": EPOCHS,
            "CHECKPOINT_INTERVAL": CHECKPOINT_INTERVAL,
            "LEARNING_RATE": LEARNING_RATE,
            "DEVICE": DEVICE
        }
        with open(os.path.join(MODEL_RESULTS_FOLDER, "training_params.json"), 'w') as f:
            json.dump(params, f, indent=4)

        self.vit_model.to(DEVICE)
        decoder_model.train()
        for epoch in range(EPOCHS):
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{EPOCHS}]', unit='batch')
            running_loss = 0.0

            for micrographs, indexes in train_dataloader:
                micrographs = micrographs.to(DEVICE)
                optimizer.zero_grad()

                latent_micrographs = self.vit_model(micrographs)

                outputs = decoder_model(latent_micrographs)
                loss = criterion(outputs, micrographs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                epoch_bar.set_postfix(loss=loss.item())
                epoch_bar.update(1)

            avg_loss = running_loss / len(train_dataloader)
            epoch_bar.set_postfix(loss=avg_loss)
            epoch_bar.close()
            # Save running loss to log file
            with open(os.path.join(MODEL_RESULTS_FOLDER, "loss.txt"), 'a') as f:
                f.write(f"{epoch + 1},{avg_loss}\n")
            # Save checkpoint
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                torch.save(decoder_model.state_dict(), os.path.join(MODEL_RESULTS_FOLDER,
                                                                    f'checkpoints/checkpoint_epoch_{epoch + 1}.pth'))

        decoder_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, (micrographs, indexes) in enumerate(test_dataloader):
                micrographs = micrographs.to(DEVICE)
                latent_micrographs = self.vit_model(micrographs)
                outputs = decoder_model(latent_micrographs)
                loss = criterion(outputs, micrographs)
                total_loss += loss.item()

                if i < EVAL_EXAMPLES:
                    compare_images(micrographs[0], outputs[0], f"example_{i}.png", MODEL_RESULTS_FOLDER,
                                   "ground truth", "prediction")

        avg_test_loss = total_loss / len(test_dataloader)
        print(f"Average test loss: {avg_test_loss}")
        with open(os.path.join(MODEL_RESULTS_FOLDER, "test_loss.txt"), 'w') as f:
            f.write(f"Average test loss: {avg_test_loss}\n")


if __name__ == '__main__':
    unittest.main()
