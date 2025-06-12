import torch
import json
import os

from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates


def prepare_dataloaders(dataset, validation_dataset, batch_size):
    """
    :param dataset: instance of torch dataset class
    :param validation_dataset: instance of torch dataset class
    :param batch_size: number determining batch size
    :return: train_dataloader, test_dataloader
    """

    # We make our custom collate function here so that the class can handle variable length target coordinates
    def collate_fn(batch):
        micrographs, heatmaps, target_coords_list, debug_tuple = zip(*batch)
        micrographs = torch.stack(micrographs)
        heatmaps = torch.stack(heatmaps)
        targets_list = list(target_coords_list)

        return micrographs, heatmaps, targets_list, debug_tuple

    # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every
    # batch needs to be of size args.batch_size
    train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset, batch_size=1, drop_last=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, drop_last=True, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, validation_dataloader
