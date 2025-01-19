import argparse
import mrcfile
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
from torchvision.models import VisionTransformer


def main():
    parser = argparse.ArgumentParser()
    # General

    # Training
    parser.add_argument("--dataset_path", type=str, default="../dataset/10017/micrographs", help="")
    parser.add_argument("--no_micrographs", type=int, default=5, help="")
    parser.add_argument("--result_dir", type=str, default="../media/example_micrographs", help="")

    args = parser.parse_args()

    # Ensure result_dir exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Pick first 10 micrographs in folder
    with os.scandir(args.dataset_path) as entries:
        micrograph_path_list = [os.path.join(args.dataset_path, entry.name)
                                for entry in islice(entries, args.no_micrographs) if entry.is_file()]

    # Save them as figures
    for i, micrograph_path in enumerate(micrograph_path_list):
        with mrcfile.open(micrograph_path) as mrc:
            data = mrc.data

        plt.imshow(data, cmap="gray")
        plt.axis('off')
        plt.savefig(args.result_dir + "/example" + str(i))


if __name__ == "__main__":
    main()
