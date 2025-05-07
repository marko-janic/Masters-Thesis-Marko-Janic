import os
import datetime
import torch

import numpy as np

from tqdm import tqdm

from postprocess import _get_max_preds
from train import create_heatmaps_from_targets, find_optimal_assignment_heatmaps
# Local imports
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_predictions_with_ground_truth, compare_heatmaps_with_ground_truth, compare_heatmaps
from vit_model import get_encoded_image


def evaluate(args, model, vit_model, vit_image_processor, dataset, test_dataloader, criterion, example_predictions):
    """

    :param args:
    :param model:
    :param vit_model:
    :param vit_image_processor:
    :param dataset:
    :param test_dataloader:
    :param criterion:
    :param example_predictions:
    :return:
    """
    result_dir = os.path.join(args.result_dir, f'evaluation_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    create_folder_if_missing(result_dir)

    model.eval()
    criterion.eval()
    running_loss = 0.0

    example_counter = 0
    with torch.no_grad():
        for micrographs, index in tqdm(test_dataloader, desc="Evaluating"):
            model.eval()
            criterion.eval()

            targets = dataset.get_targets_from_target_indexes(index, args.device)
            target_heatmaps = create_heatmaps_from_targets(targets, num_predictions=args.num_particles,
                                                           device=args.device)
            encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor)
            latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
            latent_micrographs = latent_micrographs.permute(0, 2, 1).reshape(1, args.latent_dim, 14, 14)

            outputs = model(latent_micrographs)

            assignments = find_optimal_assignment_heatmaps(outputs["heatmaps"], target_heatmaps, criterion)
            reordered_target_heatmaps = torch.zeros_like(target_heatmaps)
            for batch_idx, (row_ind, col_ind) in enumerate(assignments):
                reordered_target_heatmaps[batch_idx] = target_heatmaps[batch_idx, col_ind]

            losses = criterion(outputs["heatmaps"], target_heatmaps)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                pred_coords, scores = _get_max_preds(outputs["heatmaps"])
                particle_width_height_columns = torch.full((pred_coords.shape[0], pred_coords.shape[1], 2), 80/224,  # TODO: This needs to be adjusted if you wanna do different particle heights and widths
                                                           device=pred_coords.device)
                pred_coords = torch.cat([pred_coords, particle_width_height_columns], dim=2)

                # Filter pred_coords based on scores > 0.8 while preserving batch dimensions
                pred_coords = torch.where(scores > 0.7, pred_coords, torch.zeros_like(pred_coords))

                pred_coords = transform_coords_to_pixel_coords(224, 224, pred_coords)

                ground_truth = transform_coords_to_pixel_coords(224, 224,
                                                                targets[0]['boxes'][:, :4].unsqueeze(0))

                compare_heatmaps_with_ground_truth(micrograph=micrographs[0].cpu()/255,
                                                   particle_locations=ground_truth[0],
                                                   heatmaps=outputs["heatmaps"][0],
                                                   heatmaps_title="Model output",
                                                   result_folder_name=f"model_to_ground_truth_heatmaps_comparison_"
                                                                      f"{example_counter}",
                                                   result_dir=result_dir)

                compare_heatmaps(heatmaps_gt=reordered_target_heatmaps[0],
                                 heatmaps_pred=outputs["heatmaps"][0],
                                 result_folder_name=f"model_heatmaps_vs_target_heatmaps_{example_counter}",
                                 result_dir=result_dir)

                compare_predictions_with_ground_truth(
                    image_tensor=micrographs[0].cpu()/255,
                    ground_truth=ground_truth[0],
                    predictions=pred_coords[0],
                    object_type="output_box",
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
