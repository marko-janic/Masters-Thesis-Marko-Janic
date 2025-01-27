import argparse
import mrcfile
import os
import random
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mrcfile as mrc
import pandas as pd

from itertools import islice
from torchvision.models import VisionTransformer
from tqdm import tqdm

# Local imports
from dataset import create_sub_micrographs, get_particle_locations_from_coordinates, ShrecDataset
from plotting import save_image_with_bounding_boxes
from utils import print_separator

warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos


def shrec_dataset_example(result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    dataset = ShrecDataset(dataset_path="../dataset/shrec21_full_dataset/")

    print("Shape of data in ShrecDataset: ", dataset.sub_micrographs.shape)
    print("Head: ")
    print(dataset.sub_micrographs.head(3))
    print("Getting one specific sub micrograph: ")
    sub_micrograph_entry = dataset.__getitem__(0)
    sub_micrograph = sub_micrograph_entry.iloc[0]
    coordinates = sub_micrograph_entry.iloc[1]
    print(sub_micrograph_entry)

    selected_particles = get_particle_locations_from_coordinates(coordinates, dataset.sub_micrograph_size,
                                                                 dataset.particle_locations)
    #print("Selected particles: ")
    #print(selected_particles)

    save_image_with_bounding_boxes(dataset.micrograph, dataset.particle_locations, 4, result_dir,
                                   "test_all")
    #dataset.micrograph[412:512, 0:100] = 0
    #fig, ax = plt.subplots(1)
    #ax.imshow(dataset.micrograph, cmap='gray')
    #box_size = 6
    #rect = patches.Rectangle((0, 0), box_size, box_size, linewidth=1, edgecolor='r', facecolor='none')
    #ax.add_patch(rect)
    #plt.savefig("TEST")

    save_image_with_bounding_boxes(sub_micrograph, selected_particles, 5, result_dir, "Test0")


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

    sub_micrographs = create_sub_micrographs(micrograph, crop_size, 5)

    for i in tqdm(range(len(sub_micrographs)), desc="Saving sub micrographs"):
        sub_micrograph = sub_micrographs.iloc[i]["sub_micrograph"]
        coordinates = sub_micrographs.iloc[i]["top_left_coordinates"]

        # Save them as figures
        plt.title(f'Top left point: {coordinates[0]}, {coordinates[1]}')
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
    print_separator(label="Loading Data")
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
                                     delim_whitespace=True, names=columns).drop(columns=['Z', 'rotation_Z1',
                                                                                         'rotation_X', 'rotation_Z2'])
    print("Shape of particle locations: ", particle_locations.shape)

    # Example visualization of one micrograph ==========================================================================
    run_0 = False
    if run_0:
        print_separator(label="Visualizing one noisy micrograph")
        noisy_micrograph_example(noisy_projections[29], args.model_number, args.result_dir)

    # Cropping sub micrographs =========================================================================================
    run_1 = False
    if run_1:
        print_separator(label="Creating and saving sub micrographs")
        crop_sub_micrographs_example(micrograph=micrograph, crop_size=224,
                                     result_dir=os.path.join(args.result_dir,
                                                             f'sub_micrograph_cropping_model_{args.model_number}'),
                                     model_number=args.model_number, particle_locations=particle_locations)

    # Grandmodel micrograph ============================================================================================
    run_2 = False
    if run_2:
        print_separator(label="Checking and saving grandmodel with summed dimensions")
        grandmodel_dimension_summing_example(dataset_path=args.dataset_path, model_number=args.model_number,
                                             result_dir=os.path.join(args.result_dir,
                                                                     f'grandmodel_dimension_summing_examples_model_'
                                                                     f'{args.model_number}'))

    # Dataset implementation testing ===================================================================================
    run_3 = True
    if run_3:
        print_separator(label="Checking implementation of ShrecDataset class")
        shrec_dataset_example(os.path.join(args.result_dir, f'dataset_testing_model_{args.model_number}'))


if __name__ == "__main__":
    main()
