import argparse
import mrcfile
import torch
import types

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import vit_b_16, VisionTransformer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from model import ParticlePicker
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from loss import ParticlePickingLoss


# We use this model to override the normal implementation since we don't want the classification head:
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289
def get_latent_representation(self, x: torch.Tensor):
    # Process input
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    # Pass through encoder
    x = self.encoder(x)

    # Return the class token representation (latent)
    latent_representation = x[:, 0]
    return latent_representation


def main():
    # Arguments ========================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_particles", type=int, default=10,
                        help="Number of particles that the model outputs as predictions")

    args = parser.parse_args()

    # ==================================================================================================================
    vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
    vit_model.eval()
    # Here we replace the method of the class to use our own one that doesn't use the classification head.
    vit_model.forward = types.MethodType(get_latent_representation, vit_model)

    # Training =========================================================================================================
    model = ParticlePicker(args.latent_dim, args.num_particles)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = ParticlePickingLoss()
    dataset = ShrecDataset(16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # Loading bar for the outer loop (epochs)
        epoch_bar = tqdm(range(len(dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

        for sub_micrographs, coordinate_tl_list in dataloader:
            particle_locations_list = []

            # Loading bar for the inner loop (batches within each epoch)
            for coordinate_tl in coordinate_tl_list:
                particle_locations = get_particle_locations_from_coordinates(coordinate_tl,
                                                                             dataset.sub_micrograph_size,
                                                                             dataset.particle_locations)
                particle_locations_list.append(particle_locations[['X', 'Y']].to_numpy())

            latent_sub_micrographs = vit_model(sub_micrographs)
            predictions = model(latent_sub_micrographs)

            loss = loss_fn(particle_locations_list, predictions)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            epoch_bar.set_postfix(loss=running_loss / (epoch_bar.n + 1))  # Update the postfix with the running loss
            epoch_bar.update(1)  # Update the progress bar after each batch

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}')


if __name__ == "__main__":
    main()
