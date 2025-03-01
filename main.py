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
from dataset import ShrecDataset, DummyDataset, get_particle_locations_from_coordinates
from loss import SetCriterion, build
from evaluate import evaluate
from train import prepare_targets_for_loss, prepare_outputs_for_loss, compute_losses


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
    parser.add_argument("--dataset", type=str, default="dummy",
                        help="Which dataset to use for running the program: dummy, shrec")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the program in: train, eval")
    parser.add_argument("--existing_result_folder", type=str, default="experiment_24-02-2025_18-45-59",
                        help="Path to existing result folder to load model from.")

    # Experiment Results
    parser.add_argument("--result_dir", type=str,
                        default=f'experiments/experiment_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}',
                        help="Directory to save results to")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.01, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save model checkpoint every checkpoint_interval epochs")
    parser.add_argument("--loss_log_file", type=str, default="loss_log.txt",
                        help="File to save running loss for each epoch")
    parser.add_argument("--sampling_points", type=int, default=16,
                        help="Number of sampling points when creating sub micrographs: sampling_points*sampling_points "
                             "equals total number of sub micrographs")
    parser.add_argument("--train_eval_split", type=float, default=0.9,
                        help="Ratio of training to evaluation split. 0.9 means that 90% of the data is used for "
                             "training and 10% for evaluation")

    # Data
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    # TODO: add checker for when num_particles is somehow less than the ground truth ones in the sub micrograph
    parser.add_argument("--num_particles", type=int, default=100,
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
    loss_log_path = ""
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
    if args.dataset == "dummy":
        dataset = DummyDataset(dataset_size=50)
    elif args.dataset == "shrec":
        dataset = ShrecDataset(args.sampling_points)
    else:
        dataset = ShrecDataset(args.sampling_points)

    train_size = int(args.train_eval_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if args.dataset == "dummy":
        model = ParticlePicker(args.latent_dim, args.num_particles, dataset.image_width,
                               dataset.image_height)
    elif args.dataset == "shrec":
        model = ParticlePicker(args.latent_dim, args.num_particles, dataset.sub_micrograph_size,
                               dataset.sub_micrograph_size)
    else:
        model = ParticlePicker(args.latent_dim, args.num_particles, dataset.sub_micrograph_size,
                               dataset.sub_micrograph_size)

    model.to(args.device)
    criterion, postprocessors = build(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO: add weight decay

    # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every
    # batch needs to be of size args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    if args.mode == "train":
        # Save untrained checkpoint for debugging purposes
        torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_untrained.pth'))

        for epoch in range(args.epochs):
            model.train()
            criterion.train()

            running_loss = 0.0
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')
            for micrographs, index in train_dataloader:
                losses, outputs, targets = compute_losses(args, index, dataset, model, vit_model, micrographs, criterion)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

                epoch_bar.set_postfix(loss=losses.item())
                epoch_bar.update(1)
            epoch_bar.close()

            avg_loss = running_loss / len(train_dataloader)

            # Save running loss to log file
            with open(loss_log_path, 'a') as f:
                f.write(f"{epoch + 1},{avg_loss}\n")

            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.result_dir,
                                                            f'checkpoints/checkpoint_epoch_{epoch + 1}.pth'))

        # Save final checkpoint
        torch.save(model.state_dict(), os.path.join(args.result_dir, 'checkpoint_final.pth'))

    if args.mode == "eval":
        checkpoint_path = os.path.join(args.result_dir, 'checkpoint_final.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        evaluate(args=args, criterion=criterion, vit_model=vit_model, model=model,
                 dataset=dataset, test_dataloader=test_dataloader, example_predictions=4, postprocessors=postprocessors)


if __name__ == "__main__":
    main()
