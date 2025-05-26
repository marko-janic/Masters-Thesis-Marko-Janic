import os
import datetime
import torch

import numpy as np

from tqdm import tqdm

from postprocess import _get_max_preds
from train import create_heatmaps_from_targets, find_optimal_assignment_heatmaps, get_targets
# Local imports
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_predictions_with_ground_truth, compare_heatmaps_with_ground_truth, compare_heatmaps
from vit_model import get_encoded_image


def read_arguments_file(filepath):
    args_dict = {}
    with open(filepath, 'r') as f_file:
        for line in f_file:
            if ':' in line:
                key_val, value = line.strip().split(':', 1)
                args_dict[key_val.strip()] = value.strip()
    return args_dict


def evaluate(args, model, vit_model, vit_image_processor, dataset, test_dataloader, criterion, example_predictions):
    """

    :param args: Needs:
        one_heatmap
        result_dir
        dataset
        num_particles
        device
        latent_dim
        vit_input_size
        particle_height
        particle_width
        prediction_threshold
    :param model:
    :param vit_model:
    :param vit_image_processor:
    :param dataset:
    :param test_dataloader:
    :param criterion:
    :param example_predictions:
    :return:
    """
    result_dir = os.path.join(args.result_dir, f'evaluation_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    create_folder_if_missing(result_dir)
    with open(os.path.join(result_dir, 'arguments.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    experiment_result_dir = args.result_dir

    # --- Begin argument consistency check ---
    allowed_missing_or_different_keys = [
        # Add keys here that are allowed to be missing in evaluation args
        # Example:
        # "some_optional_key",
        "loss_log_path",
        "prediction_threshold",
        "checkpoint_interval",
        "learning_rate",
        "epochs",
        "device",
        "existing_result_folder",
        "mode",
        "config",
        "result_dir_appended_name"
    ]
    arguments_file_evaluation = os.path.join(result_dir, 'arguments.txt')
    arguments_file_experiment = os.path.join(experiment_result_dir, 'arguments.txt')
    if not os.path.exists(arguments_file_experiment):
        raise FileNotFoundError(f"arguments.txt not found in experiment_result_dir: {experiment_result_dir}")

    args_evaluation = read_arguments_file(arguments_file_evaluation)
    args_experiment = read_arguments_file(arguments_file_experiment)

    with open(arguments_file_evaluation, 'a') as log_f:
        for key in args_evaluation:
            if key in allowed_missing_or_different_keys:
                continue  # skip allowed keys for both missing and difference
            if key in args_experiment:
                if args_evaluation[key] != args_experiment[key]:
                    msg = (f"Argument '{key}' differs:\n"
                           f"  evaluation_result_dir value: {args_evaluation[key]}\n"
                           f"  experiment_result_dir value: {args_experiment[key]}")
                    print(msg)
                    log_f.write(f"[DIFFERENCE] {msg}\n")
            else:
                msg = f"Argument '{key}' is only present in evaluation arguments.txt"
                print(msg)
                log_f.write(f"[MISSING] {msg}\n")
        # Check for keys in experiment args that are missing in evaluation args, unless allowed
        for key in args_experiment:
            if key in allowed_missing_or_different_keys:
                continue  # skip allowed keys for both missing and difference
            if key not in args_evaluation:
                msg = f"Argument '{key}' is present in experiment arguments.txt but missing in evaluation"
                print(msg)
                log_f.write(f"[MISSING_IN_EVAL] {msg}\n")
    # --- End argument consistency check ---

    model.eval()
    criterion.eval()
    running_loss = 0.0
    running_pixel_loss = 0  # average amount of pixels that the predictions are off
    running_missing_predictions = 0
    running_extra_predictions = 0

    example_counter = 0
    with torch.no_grad():
        for micrographs, index in tqdm(test_dataloader, desc="Evaluating"):
            model.eval()
            criterion.eval()

            target_heatmaps, targets = get_targets(args=args, dataset=dataset, index=index)

            encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor)
            # the 1: is because we don't need the class token
            latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
            # TODO: technically you could adjust the 14, 14 to be calculated but its unnecessary as long as you don't
            #  change the vit input size
            latent_micrographs = latent_micrographs.permute(0, 2, 1).reshape(1, args.latent_dim, 14, 14)

            outputs = model(latent_micrographs)

            extra_missed_predictions = 0
            # We use this to compute the average pixels off so to speak, we set it to -1 to indicate no prediction
            padded_target_boxes = torch.full((1, args.num_particles, 2), -1.0, dtype=torch.float32)
            for i in range(len(targets)):
                indices = len(targets[i]["boxes"])
                if indices > args.num_particles:
                    extra_missed_predictions = extra_missed_predictions + indices - args.num_particles
                    indices = args.num_particles
                padded_target_boxes[i, :indices, :] = targets[i]["boxes"][:indices, :2]
            assignments = find_optimal_assignment_heatmaps(outputs["heatmaps"], target_heatmaps, criterion)
            # We put -1 to signify there is no prediction here
            reordered_padded_target_boxes = torch.full_like(padded_target_boxes, -1.0, dtype=torch.float32)
            reordered_target_heatmaps = torch.zeros_like(target_heatmaps)
            for batch_idx, (row_ind, col_ind) in enumerate(assignments):
                reordered_target_heatmaps[batch_idx] = target_heatmaps[batch_idx, col_ind]
                reordered_padded_target_boxes[batch_idx] = padded_target_boxes[batch_idx, col_ind]
            reordered_padded_target_boxes = reordered_padded_target_boxes * args.vit_input_size  # back to pixel coords

            losses = criterion(outputs["heatmaps"], reordered_target_heatmaps)
            running_loss += losses.item()

            pred_coords, pred_scores = _get_max_preds(outputs["heatmaps"])
            particle_width_height_columns = torch.full((pred_coords.shape[0], pred_coords.shape[1], 2),
                                                       args.particle_width / args.vit_input_size,
                                                       device=pred_coords.device)
            pred_coords = torch.cat([pred_coords, particle_width_height_columns], dim=2)
            # Filter pred_coords based on scores > threshold while preserving batch dimensions
            pred_coords = torch.where(pred_scores > args.prediction_threshold, pred_coords,
                                      torch.zeros_like(pred_coords))
            pred_coords = transform_coords_to_pixel_coords(args.vit_input_size, args.vit_input_size, pred_coords)

            pixel_difference = pred_coords[:, :, :2] - reordered_padded_target_boxes

            target_exists = reordered_padded_target_boxes[:, :, 0] >= 0
            pred_exists = (pred_coords[:, :, 2] > 0) & (pred_coords[:, :, 3] > 0)

            # Missed predictions: target exists but no prediction
            missed_predictions = (target_exists & ~pred_exists).sum().item() + extra_missed_predictions

            # Extra predictions: prediction exists but no target
            extra_predictions = (~target_exists & pred_exists).sum().item()

            # Correct predictions: both exist
            correct_predictions_mask = target_exists & pred_exists
            if correct_predictions_mask.sum() > 0:
                correct_pixel_diffs = pixel_difference[correct_predictions_mask]
                avg_pixels_off = correct_pixel_diffs.norm(dim=1).mean().item()
            else:
                avg_pixels_off = 0.0

            running_missing_predictions += missed_predictions
            running_extra_predictions += extra_predictions
            running_pixel_loss += avg_pixels_off

            if example_counter < example_predictions:
                ground_truth = transform_coords_to_pixel_coords(args.vit_input_size, args.vit_input_size,
                                                                targets[0]['boxes'][:, :4].unsqueeze(0))

                compare_heatmaps_with_ground_truth(micrograph=micrographs[0].cpu(),
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
                    image_tensor=micrographs[0].cpu(),
                    ground_truth=ground_truth[0],
                    predictions=pred_coords[0],
                    object_type="output_box",
                    object_parameters={"box_width": args.particle_width, "box_height": args.particle_height},
                    result_dir=result_dir,
                    file_name=f'example_{example_counter}.png',
                    figure_title=f"Avg pixels off: {avg_pixels_off}, missing: {missed_predictions}, "
                                 f"extra: {extra_predictions}"
                )
                example_counter += 1

    avg_loss = running_loss / len(test_dataloader)
    avg_pixel_loss = running_pixel_loss / len(test_dataloader)
    avg_missed_predictions = running_missing_predictions / len(test_dataloader)
    avg_extra_predictions = running_extra_predictions / len(test_dataloader)
    print(f"Average evaluation loss: {avg_loss}")
    print(f"Average pixel loss: {avg_pixel_loss}")

    # Logging the running loss to a txt file
    log_file_path = os.path.join(result_dir, "evaluation_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Average evaluation loss: {avg_loss}\n")
        log_file.write(f"Average pixel loss: {avg_pixel_loss}\n")
        log_file.write(f"Average number of missed predictions: {avg_missed_predictions}\n")
        log_file.write(f"Average number of extra predictions: {avg_extra_predictions}\n")
        log_file.write(f"prediction_threshold: {args.prediction_threshold}")
