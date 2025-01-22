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

# Local imports
from model import ParticlePicker


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
    parser.add_argument("--batch_size", type=int, default=32, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_particles", type=int, default=50,
                        help="Number of particles that the model outputs as predictions")

    args = parser.parse_args()

    # ==================================================================================================================
    vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
    vit_model.eval()
    # Here we replace the method of the class to use our own one that doesn't use the classification head.
    vit_model.forward = types.MethodType(get_latent_representation, vit_model)

    image = torch.rand(args.batch_size, 3, 224, 224)

    output = vit_model(image)

    print("Default patch size: ", vit_model.patch_size)
    print("Output shape after ViT: ", output.shape)

    ## Training ========================================================================================================
    #model = ParticlePicker(args.latent_dim, args.num_particles)
    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #for epoch in range(args.epochs):
    #    model.train()
    #    optimizer.zero_grad()
    #    predictions = "placeholder"
    #    loss = criterion()
    #    loss.backward()
    #    optimizer.step()
    #    print("Epoch " + str())


if __name__ == "__main__":
    main()
