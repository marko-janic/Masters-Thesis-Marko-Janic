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


def get_latent_representation(self, x: torch.Tensor):
    """
    We use this model to override the normal implementation since we don't want the classification head:
    https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289
    """
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
    # Program Arguments
    parser.add_argument("--mode", type=str, default="eval", help="Mode to run the program in: train, eval")
    parser.add_argument("--existing_result_folder", type=str, default="experiment_18-02-2025_11-38-47", help="Path to existing result folder to load model from.")

    # Experiment Results
    parser.add_argument("--result_dir", type=str, default=f'experiments/experiment_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}', help="Directory to save results to")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save model checkpoint every checkpoint_interval epochs")
    parser.add_argument("--loss_log_file", type=str, default="loss_log.txt", help="File to save running loss for each epoch")
    parser.add_argument("--sampling_points", type=int, default=16, help="Number of sampling points when creating sub micrographs: sampling_points*sampling_points equals total number of sub micrographs")
    parser.add_argument("--train_eval_split", type=float, default=0.9, help="Ratio of training to evaluation split. 0.9 means that 90% of the data is used for training and 10% for evaluation")

    # Data
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_particles", type=int, default=500,  # TODO: add checker for when num_particles is somehow less than the ground truth ones in the sub micrograph
                        help="Number of particles that the model outputs as predictions")

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    args = parser.parse_args()

    # Create necessary folders if not present ==========================================================================
    if args.existing_result_folder is not None and args.mode == "eval":
        args.result_dir = os.path.join('experiments', args.existing_result_folder)

    create_folder_if_missing(args.result_dir)
    create_folder_if_missing(os.path.join(args.result_dir, 'checkpoints'))

    # Save Training information into file ==============================================================================
    if args.mode != "eval":
        with open(os.path.join(args.result_dir, 'arguments.txt'), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

    # Initialize loss log file =========================================================================================
    if args.mode != "eval":
        loss_log_path = os.path.join(args.result_dir, args.loss_log_file)
        with open(loss_log_path, 'w') as f:
            f.write("epoch,running_loss\n")

    # ==================================================================================================================
    vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
    vit_model.eval()
    # Here we replace the method of the class to use our own one that doesn't use the classification head.
    vit_model.forward = types.MethodType(get_latent_representation, vit_model)

    # Training =========================================================================================================
    model = ParticlePicker(args.latent_dim, args.num_particles)
    model.to(args.device)
    criterion = build(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO: add weight decay

    dataset = ShrecDataset(args.sampling_points)
    train_size = int(args.train_eval_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)  # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every batch needs to be of size args.batch_size
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    if args.mode == "train":
        for epoch in range(args.epochs):
            model.train()
            criterion.train()
            running_loss = 0.0
            # Loading bar for the outer loop (epochs)
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

            for sub_micrographs, coordinate_tl_list in train_dataloader:
                targets = []
                for index, coordinate_tl in enumerate(coordinate_tl_list):
                    particle_locations = get_particle_locations_from_coordinates(coordinate_tl,
                                                                                 dataset.sub_micrograph_size,
                                                                                 dataset.particle_locations)
                    # We do this so that it fits into the loss function given by cryo transformer
                    boxes = torch.tensor(particle_locations[['X', 'Y']].values)
                    zero_columns = torch.ones((boxes.shape[0], 2)) * 5  # TODO: remove this 0.01, this is a hack because I cant have the box thing to be 0
                    boxes = torch.cat((boxes, zero_columns), dim=1)
                    labels = torch.ones(boxes.shape[0])
                    orig_size = torch.tensor([dataset.sub_micrograph_size, dataset.sub_micrograph_size])
                    size = torch.tensor([dataset.sub_micrograph_size, dataset.sub_micrograph_size])

                    # Pad boxes to 500 entries (padded with 0, 0 for coordinates and 0.01 for other values)
                    pad_size = args.num_particles - boxes.shape[0]
                    if pad_size > 0:
                        padding = torch.zeros(pad_size, 4)  # Pad with 0, 0, 0, 0 (4 elements for each box)
                        boxes = torch.cat((boxes, padding), dim=0)

                    # Now, pad 'labels' to a fixed size of 500 with 0s
                    label_pad_size = args.num_particles - labels.shape[0]
                    if label_pad_size > 0:
                        label_padding = torch.zeros(label_pad_size)  # Padding for labels (all 0s)
                        labels = torch.cat((labels, label_padding), dim=0)

                    target = {
                        "boxes": boxes,
                        "labels": labels,
                        "orig_size": orig_size,
                        "image_id": size,
                    }
                    targets.append(target)

                latent_sub_micrographs = vit_model(sub_micrographs)
                predictions = model(latent_sub_micrographs)

                # Again we do it to fit the cryo transformer loss
                predictions_classes = predictions[:, :, 2:4]
                predictions_coordinates = predictions[:, :, :2]
                box_sizes = torch.ones(args.batch_size, args.num_particles, 2) * 5  # TODO: remove this 0.01, this is a hack because I cant have the box thing to be 0
                predictions_coordinates = torch.cat((predictions_coordinates, box_sizes), dim=2)
                outputs = {
                    "pred_logits": predictions_classes,
                    "pred_boxes": predictions_coordinates
                }

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses

                epoch_bar.set_postfix(loss=running_loss / (epoch_bar.n + 1))  # Update the postfix with the running loss
                epoch_bar.update(1)  # Update the progress bar after each batch
            epoch_bar.close()

            # Save running loss to log file
            with open(loss_log_path, 'a') as f:
                f.write(f"{epoch + 1},{running_loss.item()}\n")  # TODO: change this since the current value is weird

            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth'))

        # Save final checkpoint
        torch.save(model.state_dict(), os.path.join(args.result_dir, 'checkpoint_final.pth'))

    if args.mode == "eval":
        checkpoint_path = os.path.join(args.result_dir, 'checkpoint_final.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        evaluate(experiment_dir=args.result_dir, criterion=criterion, vit_model=vit_model, model=model,
                 dataset=dataset, test_dataloader=test_dataloader, device=args.device, example_predictions=4)


if __name__ == "__main__":
    main()
