import os
import datetime
import torch

import numpy as np

from tqdm import tqdm

from postprocess import _get_max_preds
from train import create_heatmaps_from_targets
# Local imports
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_predictions_with_ground_truth, compare_heatmaps_with_ground_truth, compare_heatmaps
from vit_model import get_encoded_image


def evaluate(args, model, vit_model, vit_image_processor, dataset, test_dataloader, criterion, example_predictions):
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
            outputs = model(latent_micrographs.reshape((1, args.latent_dim, 14, 14)))  # TODO: don't hardcode this 14

            losses = criterion(outputs["heatmaps"], target_heatmaps)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                pred_coords, scores = _get_max_preds(outputs["heatmaps"])
                particle_width_height_columns = torch.full((pred_coords.shape[0], pred_coords.shape[1], 2), 80/224,
                                                           device=pred_coords.device)
                pred_coords = torch.cat([pred_coords, particle_width_height_columns], dim=2)

                pred_coords = transform_coords_to_pixel_coords(224, 224, pred_coords.cpu().numpy())
                ground_truth = transform_coords_to_pixel_coords(224, 224, targets[0]['boxes'][:, :4])

                compare_heatmaps_with_ground_truth(micrograph=micrographs[0].cpu(),
                                                   particle_locations=ground_truth,
                                                   heatmaps=outputs["heatmaps"][0],
                                                   heatmaps_title="Model output",
                                                   result_folder_name=f"model_to_ground_truth_heatmaps_comparison_"
                                                                      f"{example_counter}",
                                                   result_dir=result_dir)

                compare_heatmaps(heatmaps_gt=target_heatmaps[0],
                                 heatmaps_pred=outputs["heatmaps"][0],
                                 result_folder_name=f"model_heatmaps_vs_target_heatmaps_{example_counter}",
                                 result_dir=result_dir)

                compare_predictions_with_ground_truth(
                    image_tensor=micrographs[0].cpu(),
                    ground_truth=ground_truth,
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
