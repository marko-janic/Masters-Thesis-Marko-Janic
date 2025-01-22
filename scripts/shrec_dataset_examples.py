import argparse
import mrcfile
import os
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import mrcfile as mrc

from itertools import islice
from torchvision.models import VisionTransformer

# Local imports
from dataset import create_sub_micrographs

warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset/shrec21_full_dataset", help="")
    parser.add_argument("--model_number", type=int, default=2, help="")
    parser.add_argument("--no_micrographs", type=int, default=5, help="")
    parser.add_argument("--result_dir", type=str, default="../media/shrec_example_micrographs", help="")
    args = parser.parse_args()

    # Ensure result_dir exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Example visualization of one micrograph ==========================================================================
    with mrc.open(os.path.join(args.dataset_path, f'model_{args.model_number}/projections_unbinned.mrc'),
                  permissive=True) as f:
        projections = f.data

    print("Shape of projection tensor: ", projections.shape)

    # Save them as figures
    micrograph = projections[29]
    plt.imshow(projections[29], cmap="gray")
    plt.axis('off')
    plt.savefig(args.result_dir + f'/example_model_{args.model_number}')

    # Cropping sub micrographs =========================================================================================
    crop_size = 224

    crops = create_sub_micrographs(micrograph, crop_size)

    print(f"Crops shape: {crops.shape}")

    for i in range(crops.shape[0]):
        sub_micrograph = crops[i]

        # Save them as figures
        plt.imshow(sub_micrograph, cmap="gray")
        plt.axis('off')
        plt.savefig(args.result_dir + f'/example_model_{args.model_number}_sub_micrograph_{i}')


if __name__ == "__main__":
    main()
