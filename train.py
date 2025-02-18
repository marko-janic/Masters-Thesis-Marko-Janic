import argparse
import mrcfile
import torch
import types
import os
import datetime

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import vit_b_16, VisionTransformer
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Local imports
from model import ParticlePicker
from util.utils import create_folder_if_missing
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from loss import SetCriterion, build
from evaluate import evaluate


def prepare_coordinates_for_loss(args, particle_locations, index, sub_micrograph_size):
    """
    Puts coords into right format for the loss given by CryoTransformer
    :param args: Program arguments, see main.py
    :param particle_locations: Locations of particles in the sub_micrograph
    :param index: index of entry
    :param sub_micrograph_size: Size of sub_micrographs
    :return:
    """

    # We do this so that it fits into the loss function given by cryo transformer
    boxes = torch.tensor(particle_locations[['X', 'Y', 'particle_width', 'particle_height']].values)
    pass

    labels = torch.ones(boxes.shape[0])  # all the labels are 1 since we know these are real particles

    # Pad boxes to 500 entries (padded with 0, 0 for coordinates and 0.01 for other values)
    pad_size = args.num_particles - boxes.shape[0]
    if pad_size > 0:
        padding = torch.zeros(pad_size, 4)  # Pad with 0, 0, 0, 0 (4 elements for each box)
        boxes = torch.cat((boxes, padding), dim=0)
    pass

    # Now, pad 'labels' to a fixed size of 500 with 0s
    label_pad_size = args.num_particles - labels.shape[0]
    if label_pad_size > 0:
        label_padding = torch.zeros(label_pad_size)  # Padding for labels (all 0s)
        labels = torch.cat((labels, label_padding), dim=0)

    target = {
        "boxes": boxes,
        "labels": labels,
        "orig_size": torch.tensor([sub_micrograph_size, sub_micrograph_size]),
        "image_id": index,
    }
    return target


def train_model(args, model, criterion, vit_model, dataset, train_dataloader, optimizer, loss_log_path):
    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        running_loss = 0.0
        # Loading bar for the outer loop (epochs)
        epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

        losses = 0
        for sub_micrographs, coordinate_tl_list in train_dataloader:
            targets = []
            for index, coordinate_tl in enumerate(coordinate_tl_list):
                particle_locations = get_particle_locations_from_coordinates(coordinate_tl,
                                                                             dataset.sub_micrograph_size,
                                                                             dataset.particle_locations,
                                                                             dataset.particle_width,
                                                                             dataset.particle_height)
                targets.append(prepare_coordinates_for_loss(args, particle_locations, index,
                                                            dataset.sub_micrograph_size))

            latent_sub_micrographs = vit_model(sub_micrographs)
            predictions = model(latent_sub_micrographs)

            predictions_classes = predictions[:, :, 4:6]
            predictions_boxes = predictions[:, :, :4]
            outputs = {
                "pred_logits": predictions_classes,
                "pred_boxes": predictions_boxes
            }
            pass

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            # This is how CryoTransformer does it
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses

            epoch_bar.set_postfix(loss=losses)  # Update the postfix with the running loss
            epoch_bar.update(1)  # Update the progress bar after each batch
        epoch_bar.close()

        # Save running loss to log file
        with open(loss_log_path, 'a') as f:
            f.write(f"{epoch + 1},{losses}\n")

        # Save checkpoint every checkpoint interval
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.result_dir,
                                                        f'checkpoints/checkpoint_epoch_{epoch + 1}.pth'))

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(args.result_dir, 'checkpoint_final.pth'))
