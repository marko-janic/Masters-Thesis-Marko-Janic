import argparse
import mrcfile
import torch
import types
import os
import datetime
import random
import json

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
from train import prepare_targets_for_loss, prepare_outputs_for_loss
from plotting import save_image_with_bounding_object


# Set the seed for reproducibility
# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)


def get_args():
    # Arguments ========================================================================================================
    parser = argparse.ArgumentParser()
    # Program Arguments
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--dataset", type=str, default="dummy",
                        help="Which dataset to use for running the program: dummy, shrec")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the program in: train, eval")
    parser.add_argument("--existing_result_folder", type=str, default="",
                        help="Path to existing result folder to load model from.")
    parser.add_argument("--dataset_path", type=str, default="dataset/dummy_dataset/data")
    parser.add_argument("--dataset_size", type=int, default=500)
    # TODO: add checker for when num_particles is somehow less than the ground truth ones in the sub micrograph
    parser.add_argument("--num_particles", type=int, default=7,
                        help="Number of particles that the model outputs as predictions")
    parser.add_argument("--particle_width", type=int, default=80)
    parser.add_argument("--particle_height", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    # Experiment Results
    parser.add_argument("--result_dir", type=str,
                        default=f'experiments/experiment_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}_dummy_dataset',
                        help="Directory to save results to")

    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=int, default=0.01, help="Learning rate for training")
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

    # Evaluation
    parser.add_argument('--quartile_threshold', type=float, default=0.9, help='Quartile threshold')

    args = parser.parse_args()

    # Load arguments from configuration file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    return args


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

    # Return the class token representation
    return x[:, 0]


def main():
    args = get_args()

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
            f.write("epoch,average_loss\n")

    # ==================================================================================================================
    vit_model = vit_b_16(weights="IMAGENET1K_V1", progress=True)
    vit_model.eval()
    # Here we replace the method of the class to use our own one that doesn't use the classification head.
    vit_model.forward = types.MethodType(get_latent_representation, vit_model)

    # Training =========================================================================================================
    dataset = DummyDataset(dataset_size=args.dataset_size, dataset_path=args.dataset_path,
                           particle_width=args.particle_width, particle_height=args.particle_height)

    train_size = int(args.train_eval_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    model = ParticlePicker(latent_dim=args.latent_dim, num_particles=args.num_particles,
                           image_width=dataset.image_width, image_height=dataset.image_height)

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

        plotted = False
        for epoch in range(args.epochs):
            running_loss = 0.0
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

            for micrographs, index in train_dataloader:
                # TODO: add train_one_epoch function
                model.train()
                criterion.train()

                targets = []
                for target_index in index:
                    target = dataset.targets[target_index]
                    # Move target to the same device as the model
                    target = {k: v.to(args.device) for k, v in target.items() if k != "image_id"}
                    targets.append(target)

                if epoch < 1 and not plotted:
                    save_image_with_bounding_object(micrographs[0].cpu(), targets[0]['boxes'].cpu()*224, "output_box",
                                                    {}, args.result_dir, "train_test_example")

                latent_micrographs = vit_model(micrographs).to(args.device)
                predictions = model(latent_micrographs)
                outputs = prepare_outputs_for_loss(predictions)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
                 dataset=dataset, test_dataloader=test_dataloader, example_predictions=8, postprocessors=postprocessors)


if __name__ == "__main__":
    main()
