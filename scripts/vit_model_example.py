import argparse
import mrcfile
import torch
import types

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import vit_b_16, VisionTransformer
from torchvision import transforms


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
    # ==================================================================================================================
    model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
    model.eval()
    # Here we replace the method of the class to use our own one that doesn't use the classification head.
    model.forward = types.MethodType(get_latent_representation, model)

    # Transform to preprocess the image (resize, normalization, etc.)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    img = Image.open("../media/example_micrographs/example0.png")
    print("Image before transformations: ", img.size)
    img_tensor = preprocess(img).unsqueeze(0)
    img_tensor = img_tensor[:, 0:3]
    print("Input image shape: ", img_tensor.shape)

    output = model(img_tensor)

    print("Default patch size: ", model.patch_size)
    print("Output shape after ViT: ", output.shape)


if __name__ == "__main__":
    main()
