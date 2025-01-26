import argparse
import mrcfile
import os
import random
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
import mrcfile as mrc
import pandas as pd

from itertools import islice
from torchvision.models import VisionTransformer
from tqdm import tqdm

# Local imports
from dataset import create_sub_micrographs

warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos


def noisy_micrograph_example(micrograph, model_number, result_dir):
    print("Saving example noisy micrograph to ", result_dir)
    plt.imshow(micrograph, cmap="gray")
    plt.axis('off')
    plt.savefig(result_dir + f'/example_model_{model_number}')


def crop_sub_micrographs_example(micrograph, crop_size, result_dir, model_number, particle_locations):
    print("Creating sub micrographs and saving them to ", result_dir)

    # Ensure result_dir exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    crops = create_sub_micrographs(micrograph, crop_size, 3)

    print(f"Crops shape: {crops.shape}")

    for i in tqdm(range(len(crops)), desc="Saving sub micrographs"):
        sub_micrograph = crops[i]

        # Save them as figures
        plt.imshow(sub_micrograph, cmap="gray")
        plt.axis('off')
        plt.savefig(result_dir + f'/example_model_{model_number}_sub_micrograph_{i}')


def grandmodel_dimension_summing_example(dataset_path, model_number, result_dir):
    """
    Checks which dimension should be summed up to convert the gradmodel to the ground truth micrograph
    """
    print("Summing dimensions of grandmodel and saving examples to ", result_dir)

    # Ensure result_dir exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with mrc.open(os.path.join(dataset_path, f'model_{model_number}/grandmodel.mrc'), permissive=True) as f:
        model_data = f.data

    # Sum them across axis to get a 0 degree projection so to speak (we do all to see which is which
    model_data_first_dimension = np.sum(model_data, axis=0)  # This one is the correct one it seems, see the outputs
    model_data_second_dimension = np.sum(model_data, axis=1)
    model_data_third_dimension = np.sum(model_data, axis=2)

    # Save them as figures
    plt.imshow(model_data_first_dimension, cmap="gray")
    plt.axis('off')
    plt.savefig(result_dir + f'/example_model_{model_number}_grandmodel_first_dimension_collapsed')
    plt.imshow(model_data_second_dimension, cmap="gray")
    plt.axis('off')
    plt.savefig(result_dir + f'/example_model_{model_number}_grandmodel_second_dimension_collapsed')
    plt.imshow(model_data_third_dimension, cmap="gray")
    plt.axis('off')
    plt.savefig(result_dir + f'/example_model_{model_number}_grandmodel_third_dimension_collapsed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset/shrec21_full_dataset", help="")
    parser.add_argument("--model_number", type=int, default=1, help="")
    parser.add_argument("--no_micrographs", type=int, default=5, help="")
    parser.add_argument("--result_dir", type=str, default="../media/shrec_example_micrographs", help="")
    args = parser.parse_args()

    # Ensure result_dir exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Get Data =========================================================================================================
    with mrc.open(os.path.join(args.dataset_path, f'model_{args.model_number}/projections_unbinned.mrc'),
                  permissive=True) as f:
        noisy_projections = f.data
    print("Shape of tensor with noisy projection: ", noisy_projections.shape)

    with mrc.open(os.path.join(args.dataset_path, f'model_{args.model_number}/grandmodel.mrc'),
                  permissive=True) as f:
        grandmodel = f.data
    print("Shape of grandmodel tensor: ", grandmodel.shape)

    micrograph = np.sum(grandmodel, axis=0)  # We know this is correct, see function above
    print("Shape of micrograph tensor: ", micrograph.shape)

    columns = ['class', 'X', 'Y', 'Z', 'rotation_Z1', 'rotation_X', 'rotation_Z2']
    particle_locations = pd.read_csv(os.path.join(args.dataset_path,
                                                  f'model_{args.model_number}/particle_locations.txt'),
                                     delim_whitespace=True, names=columns)
    print("Shape of particle locations: ", particle_locations.shape)

    # Example visualization of one micrograph ==========================================================================
    run_0 = False
    if run_0:
        noisy_micrograph_example(noisy_projections[29], args.model_number, args.result_dir)

    # Cropping sub micrographs =========================================================================================
    run_1 = True
    if run_1:
        crop_sub_micrographs_example(micrograph=micrograph, crop_size=224,
                                     result_dir=os.path.join(args.result_dir,
                                                             f'sub_micrograph_cropping_model_{args.model_number}'),
                                     model_number=args.model_number, particle_locations=particle_locations)

    # Grandmodel micrograph ============================================================================================
    run_2 = False
    if run_2:
        grandmodel_dimension_summing_example(dataset_path=args.dataset_path, model_number=args.model_number,
                                             result_dir=os.path.join(args.result_dir,
                                                                     f'grandmodel_dimension_summing_examples_model_'
                                                                     f'{args.model_number}'))


if __name__ == "__main__":
    main()
