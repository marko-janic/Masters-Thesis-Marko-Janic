import os
import torch
import datetime
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from model import ParticlePicker
from loss import SetCriterion, build
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_predictions_with_ground_truth
from train import prepare_targets_for_loss, prepare_outputs_for_loss, compute_losses


def evaluate(args, model, vit_model, dataset, test_dataloader, criterion, example_predictions, postprocessors):
    result_dir = os.path.join(args.result_dir, f'evaluation_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    create_folder_if_missing(result_dir)

    model.eval()
    criterion.eval()
    running_loss = 0.0

    example_counter = 0
    with torch.no_grad():
        for micrographs, index in tqdm(test_dataloader, desc="Evaluating"):
            losses, outputs, targets = compute_losses(args, index, dataset, model, vit_model, micrographs, criterion)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                target_sizes = torch.tensor([(224, 224)] * 1)

                pred_coords = transform_coords_to_pixel_coords(224, 224, outputs['boxes'][:4].cpu().numpy())
                ground_truth = transform_coords_to_pixel_coords(224, 224, targets[0]['boxes'][:, :2])

                compare_predictions_with_ground_truth(
                    image_tensor=micrographs[0].cpu(),
                    ground_truth=ground_truth,
                    predictions=pred_coords,
                    object_type="box",
                    object_parameters={"box_width": 10, "box_height": 10},
                    result_dir=result_dir,
                    file_name=f'example_{example_counter}.png'
                )
                example_counter += 1

    avg_loss = running_loss / len(test_dataloader)
    print(f"Average evaluation loss: {avg_loss}")

    # Logging the running loss to a txt file
    log_file_path = os.path.join(result_dir, "evaluation_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Average evaluation loss: {avg_loss}\n")
