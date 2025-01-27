import argparse
import mrcfile
import torch
import types

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import vit_b_16, VisionTransformer
from torchvision import transforms

# Local imports
from dataset import ShrecDataset


def main():
    dataset = ShrecDataset(dataset_path="../dataset/shrec21_full_dataset/")

    print(dataset.sub_micrographs.shape)


if __name__ == "__main__":
    main()
