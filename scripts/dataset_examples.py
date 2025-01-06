import argparse
import mrcfile
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import VisionTransformer


def main():
    parser = argparse.ArgumentParser()
    # General

    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    args = parser.parse_args()

    with mrcfile.open("dataset/10017/micrographs/Falcon_2012_06_12-15_30_21_0.mrc") as mrc:
        data = mrc.data
        print(data.shape)

    # Find the maximum value
    max_value = np.max(data)

    # Find the minimum value
    min_value = np.min(data)

    print(f"Maximum value: {max_value}")
    print(f"Minimum value: {min_value}")

    plt.imshow(data, cmap="gray")
    plt.axis('off')
    plt.savefig("")


if __name__ == "__main__":
    main()
