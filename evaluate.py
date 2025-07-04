import math
import os
import datetime
import json
import napari
import torch

import numpy as np
from skimage.feature import peak_local_max
from skimage.transform import resize
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Local imports
from postprocess import _get_max_preds
from utils import create_folder_if_missing
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


def generate_heatmaps_volume(dataset, vit_model, vit_image_processor, model, model_number: int, args):
    if args.use_fbp:
        volume = dataset.grandmodel_fbp[model_number]
    else:
        volume = dataset.grandmodel[model_number]
    z_max, y_max, x_max = volume.shape

    if args.shrec_max_z > z_max:
        raise Exception(f"shrec_max_z ({args.shrec_max_z}) can't be larger than the volume z ({z_max})")

    output_heatmap_volume = torch.zeros((z_max, 256, 256)).to(args.device)  # TODO: replace this 256 with an argument

    for z in tqdm(range(args.shrec_min_z, args.shrec_max_z), desc="Generating heatmaps volume"):
        z_slice = volume[z]

        x_step_size = x_max // (math.ceil(x_max / args.vit_input_size))
        y_step_size = y_max // (math.ceil(y_max / args.vit_input_size))
        for y_index in range(math.ceil(y_max / args.vit_input_size)):
            for x_index in range(math.ceil(x_max / args.vit_input_size)):
                y_start = y_index * y_step_size
                if y_start > (y_max - args.vit_input_size):
                    y_start = y_max - args.vit_input_size
                y_end = y_start + args.vit_input_size

                x_start = x_index * x_step_size
                if x_start > (x_max - args.vit_input_size):
                    x_start = x_max - args.vit_input_size
                x_end = x_start + args.vit_input_size

                sub_micrograph = z_slice[y_start:y_end, x_start:x_end]
                if sub_micrograph.max() > sub_micrograph.min():  # We don't need to normalize if everything is 0
                    sub_micrograph = (sub_micrograph - sub_micrograph.min()) / (sub_micrograph.max() -
                                                                                sub_micrograph.min())
                sub_micrograph = sub_micrograph.repeat(3, 1, 1)

                latent_micrographs = get_encoded_image(sub_micrograph, vit_model, vit_image_processor,
                                                       device=args.device,
                                                       num_patch_embeddings=args.num_patch_embeddings)

                outputs = model(latent_micrographs)
                heatmap = outputs["heatmaps"][0, 0]

                # TODO: check if this 0.5 wonky stuff is ok here
                y_start = math.floor(y_start*0.5)
                x_start = math.floor(x_start*0.5)
                y_end = math.floor(y_end*0.5)
                x_end = math.floor(x_end*0.5)
                output_heatmap_volume[z, y_start:y_end, x_start:x_end] = torch.max(
                    heatmap, output_heatmap_volume[z, y_start:y_end, x_start:x_end])

    return output_heatmap_volume


def evaluate(args, model, vit_model, vit_image_processor, dataset, test_dataloader, criterion, example_predictions):
    """
    :param args: Needs:
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
    if args.existing_evaluation_folder == "" or args.existing_evaluation_folder is None:
        this_evaluation_result_dir = os.path.join(args.result_dir,
                                  f'evaluation_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_'
                                  f'model{args.shrec_model_number}')
        if args.volume_evaluation:
            this_evaluation_result_dir += "_volume_evaluation"
        create_folder_if_missing(this_evaluation_result_dir)
        with open(os.path.join(this_evaluation_result_dir, 'arguments.txt'), 'a') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
    else:
        this_evaluation_result_dir = os.path.join("experiments", args.existing_result_folder, args.existing_evaluation_folder)

    experiment_result_dir = args.result_dir

    # --- Begin argument consistency check ---
    allowed_missing_or_different_keys = [
        # Add keys here that are allowed to be missing or different in args when comparing eval to train args
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
        "result_dir_appended_name",
        "shrec_sampling_points",
        "result_dir",
        "batch_size",
        "prediction_threshold",
        "neighborhood_size",
        "volume_evaluation",
        "existing_evaluation_folder",
        "split_file_name",
        "missing_pred_threshold",
        "use_train_dataset_for_evaluation",
        "train_eval_split",
        "random_sub_micrographs",
        "validation_loss_log_path",
        "patience",
        "find_optimal_parameters"
    ]
    arguments_file_evaluation = os.path.join(this_evaluation_result_dir, 'arguments.txt')
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
                    if args.existing_evaluation_folder == "" or args.existing_evaluation_folder is None:
                        log_f.write(f"[DIFFERENCE] {msg}\n")
            else:
                msg = f"Argument '{key}' is only present in evaluation arguments.txt"
                print(msg)
                if args.existing_evaluation_folder == "" or args.existing_evaluation_folder is None:
                    log_f.write(f"[MISSING] {msg}\n")
        # Check for keys in experiment args that are missing in evaluation args, unless allowed
        for key in args_experiment:
            if key in allowed_missing_or_different_keys:
                continue  # skip allowed keys for both missing and difference
            if key not in args_evaluation:
                msg = f"Argument '{key}' is present in experiment arguments.txt but missing in evaluation"
                print(msg)
                if args.existing_evaluation_folder == "" or args.existing_evaluation_folder is None:
                    log_f.write(f"[MISSING_IN_EVAL] {msg}\n")
    # --- End argument consistency check ---

    model.eval()
    criterion.eval()
    running_loss = 0.0
    running_pixel_loss = 0  # average amount of pixels that the predictions are off
    running_missing_predictions = 0
    running_extra_predictions = 0
    example_counter = 0

    with (torch.no_grad()):
        # TODO: This needs to be cleaned up some more
        if not args.volume_evaluation:
            for micrographs, target_heatmaps, target_coordinates, debug_tuple in tqdm(test_dataloader,
                                                                                      desc="Evaluating"):
                model.eval()
                criterion.eval()

                latent_micrographs = get_encoded_image(micrographs, vit_model, vit_image_processor,
                                                       device=args.device,
                                                       num_patch_embeddings=args.num_patch_embeddings)
                outputs = model(latent_micrographs)

                losses = criterion(outputs["heatmaps"], target_heatmaps)
                running_loss += losses.item()

                missed_predictions = 0
                extra_predictions = 0
                avg_pixels_off = 0

                predictions = torch.from_numpy(peak_local_max(
                    outputs["heatmaps"][0, 0].cpu().numpy(), min_distance=args.neighborhood_size,
                    threshold_abs=args.prediction_threshold))
                predictions *= 2  # The heatmaps are size 112 x 112 and the actual coordinates are 224 x 224
                predictions[:, 0] = args.vit_input_size - predictions[:, 0]
                predictions = predictions[:, [1, 0]]

                # Scale back coordinates to pixel values
                target_coordinates[0][:] = target_coordinates[0][:]*args.vit_input_size

                num_predictions = predictions.shape[0]
                num_targets = target_coordinates[0].shape[0]

                # Compute pairwise distances (num_predictions, num_targets), p=2 for euclidian distance
                dists = torch.cdist(predictions.float(), target_coordinates[0][:, :2].float(), p=2).cpu().numpy()
                # Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(dists)
                # Only consider matches where both indices are valid
                matched_dists = dists[row_ind, col_ind]
                avg_pixels_off = matched_dists.mean() if len(matched_dists) > 0 else 0.0

                # Extra predictions: predictions not matched (if more predictions than targets)
                extra_predictions += max(0, num_predictions - num_targets)
                # Missing predictions: targets not matched (if more targets than predictions)
                missed_predictions += max(0, num_targets - num_predictions)

                pred_coords = predictions
                particle_width_height_columns = torch.full((pred_coords.shape[0], pred_coords.shape[1]),
                                                           args.particle_width, device=pred_coords.device)
                pred_coords = torch.cat([pred_coords, particle_width_height_columns], dim=1).unsqueeze(0)

                running_missing_predictions += missed_predictions
                running_extra_predictions += extra_predictions
                running_pixel_loss += avg_pixels_off

                if example_counter < example_predictions:
                    compare_heatmaps_with_ground_truth(micrograph=micrographs[0].cpu(),
                                                       particle_locations=target_coordinates[0],
                                                       heatmaps=outputs["heatmaps"][0].cpu(),
                                                       heatmaps_title="Model output",
                                                       result_file_name=f"model_to_ground_truth_heatmaps_comparison_"
                                                                        f"{example_counter}",
                                                       result_dir=this_evaluation_result_dir)

                    compare_heatmaps(heatmaps_gt=target_heatmaps[0].cpu(),
                                     heatmaps_pred=outputs["heatmaps"][0].cpu(),
                                     result_folder_name=f"model_heatmaps_vs_target_heatmaps_{example_counter}",
                                     result_dir=this_evaluation_result_dir)

                    compare_predictions_with_ground_truth(
                        image_tensor=micrographs[0].cpu(),
                        ground_truth=target_coordinates[0],
                        predictions=pred_coords[0],
                        object_type="output_box",
                        object_parameters={"box_width": args.particle_width, "box_height": args.particle_height},
                        result_dir=this_evaluation_result_dir,
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
            print(f"Average number of missed predictions: {avg_missed_predictions}")
            print(f"Average number of extra predictions: {avg_extra_predictions}")
            print(f"prediction_threshold: {args.prediction_threshold}")
        else:
            output_heatmaps_volumes = {}
            for model_num in args.shrec_model_number:
                output_heatmap_volume_path = os.path.join(experiment_result_dir,
                                                          f"output_heatmaps_volume_shrec_volume_{model_num}.npy")
                if os.path.exists(output_heatmap_volume_path):
                    print(f"Found shrec volume {model_num}")
                    output_heatmaps_volumes[model_num] = np.load(output_heatmap_volume_path)
                else:
                    print(f"Shrec volume {model_num} has not been generated yet. Generating now...")
                    output_heatmaps_volumes[model_num] = generate_heatmaps_volume(
                        dataset=dataset, vit_model=vit_model, vit_image_processor=vit_image_processor,
                        model_number=model_num, args=args, model=model).cpu().numpy()
                    np.save(output_heatmap_volume_path, output_heatmaps_volumes[model_num])

            target_coordinates_dict = dataset.get_particle_locations_of_models()

            if args.find_optimal_parameters:
                best_f1_score, best_prediction_threshold, best_neighborhood_size, results = find_optimal_parameters(
                    args.shrec_model_number, target_coordinates_dict, output_heatmaps_volumes,
                    args.missing_pred_threshold, dataset=dataset)
                with open(os.path.join(experiment_result_dir, "optimal_parameters.txt"), "a") as log_file:
                    log_file.write(
                        f"Calculation of optimal parameters from "
                        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        f"\nBest average f1 score: {best_f1_score}"
                        f"\nBest prediction threshold: {best_prediction_threshold}"
                        f"\nBest neighborhood size: {best_neighborhood_size},"
                        f"\nArray with all checked results: \n"
                        f"{results}")
                with open(os.path.join(this_evaluation_result_dir, "grid_search_parameters.json"), "w") as f:
                    json.dump(results, f)

            avg_f1_score = 0
            avg_precision = 0
            avg_recall = 0
            avg_avg_pixels_off = 0
            for model_num in args.shrec_model_number:
                evaluation_dict = evaluate_predictions(
                    target_coordinates_dict=target_coordinates_dict, output_heatmaps_volumes=output_heatmaps_volumes,
                    model_num=model_num, missing_pred_threshold=args.missing_pred_threshold,
                    prediction_threshold=args.prediction_threshold, neighborhood_size=args.neighborhood_size,
                    dataset=dataset)
                avg_f1_score += evaluation_dict['f1_score']
                avg_precision += evaluation_dict['precision']
                avg_recall += evaluation_dict['recall']
                avg_avg_pixels_off += evaluation_dict['avg_pixel_loss']

                output = (f"\nEvaluation: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for shrec volume"
                          f"{model_num}"
                          f"\nTotal number of predictions: {evaluation_dict['num_preds']}"
                          f"\nTotal number of targets: {evaluation_dict['num_targets']}"
                          f"\nAverage pixel loss: {evaluation_dict['avg_pixel_loss']}"
                          f"\nMissed predictions: {evaluation_dict['avg_missed_predictions']}"
                          f"\nExtra predictions: {evaluation_dict['avg_extra_predictions']}"
                          f"\nCorrect predictions: {evaluation_dict['correct_predictions']}"
                          f"\nprediction_threshold: {args.prediction_threshold}"
                          f"\nneighborhood_size: {args.neighborhood_size}"
                          f"\nmissing_pred_threshold: {args.missing_pred_threshold}"
                          f"\nF1 Score: {evaluation_dict['f1_score']}, precision: {evaluation_dict['precision']}, "
                          f"recall: {evaluation_dict['recall']}"
                          f"\n")
                print(output)

                # Logging the running loss to a txt file
                log_file_path = os.path.join(this_evaluation_result_dir, "evaluation_log.txt")
                with open(log_file_path, "a") as log_file:
                    log_file.write(output)
            avg_f1_score = avg_f1_score / len(args.shrec_model_number)
            avg_precision = avg_precision / len(args.shrec_model_number)
            avg_recall = avg_recall / len(args.shrec_model_number)
            avg_avg_pixels_off = avg_avg_pixels_off / len(args.shrec_model_number)

            # Logging the running loss to a txt file
            log_file_path = os.path.join(this_evaluation_result_dir, "evaluation_log.txt")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\nAverage f1 score of last {len(args.shrec_model_number)} evaluations "
                               f"with models {args.shrec_model_number}: {avg_f1_score}"
                               f"\nAverage f1 score, precision and recal: {avg_f1_score},{avg_precision},{avg_recall}"
                               f"\nAverage pixels off: {avg_avg_pixels_off}")


def find_optimal_parameters(model_numbers, target_coordinates_dict, output_heatmaps_volumes,
                            missing_pred_threshold, dataset):
    """
    Performs grid search to find optimal parameters
    :param model_numbers: list of model numbers to check this on
    :param target_coordinates_dict:
    :param output_heatmaps_volumes:
    :param missing_pred_threshold
    :return: best_f1_score, best_prediction_threshold, best_neighborhood_size, results
        where results is a dictionary with the lists of parameters and avg_f1_scores that were checked
    """
    prediction_threshold_range = torch.arange(0.1, 0.7, 0.025)
    neighborhood_size_range = torch.arange(1, 10, 1)
    best_f1_score = 0
    best_prediction_threshold = 0
    best_neighborhood_size = 0

    results = {"prediction_thresholds": [], "neighborhood_sizes": [], "avg_f1_scores": [], "avg_precision": [],
               "avg_recall": []}
    for prediction_threshold in tqdm(prediction_threshold_range, desc="Grid search prediction threshold"):
        for neighborhood_size in tqdm(neighborhood_size_range, desc="Grid search neighborhood size"):
            avg_f1_score = 0
            avg_precision = 0
            avg_recall = 0
            for model_num in model_numbers:
                evaluation_dict = evaluate_predictions(
                    target_coordinates_dict=target_coordinates_dict, output_heatmaps_volumes=output_heatmaps_volumes,
                    model_num=model_num, missing_pred_threshold=missing_pred_threshold,
                    prediction_threshold=prediction_threshold.item(), neighborhood_size=neighborhood_size.item(),
                    dataset=dataset)
                avg_f1_score += evaluation_dict["f1_score"]
                avg_recall += evaluation_dict["recall"]
                avg_precision += evaluation_dict["precision"]
            avg_f1_score = avg_f1_score / len(model_numbers)
            avg_recall = avg_recall / len(model_numbers)
            avg_precision = avg_precision / len(model_numbers)

            results["prediction_thresholds"].append(prediction_threshold.item())
            results["neighborhood_sizes"].append(neighborhood_size.item())
            results["avg_f1_scores"].append(avg_f1_score)
            results["avg_precision"].append(avg_precision)
            results["avg_recall"].append(avg_recall)

            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                best_prediction_threshold = prediction_threshold
                best_neighborhood_size = neighborhood_size

    print(f"Best average f1 score: {best_f1_score} with pred_threshold: {best_prediction_threshold} "
          f"and neighborhood_size: {best_neighborhood_size}")
    return best_f1_score, best_prediction_threshold, best_neighborhood_size, results


def evaluate_predictions(target_coordinates_dict, output_heatmaps_volumes, model_num, missing_pred_threshold,
                         prediction_threshold, neighborhood_size, dataset):
    this_output_heatmaps_volume = output_heatmaps_volumes[model_num]

    coordinates = torch.from_numpy(peak_local_max(this_output_heatmaps_volume,
                                                  min_distance=neighborhood_size,
                                                  threshold_abs=prediction_threshold))
    coordinates[:, 1:] = coordinates[:, 1:] * 2  # Scale them since heatmaps are half the size

    # target_heatmap_volume = dataset.heatmaps_volume[model_num]
    # viewer = napari.Viewer()
    # viewer.add_points(coordinates, size=5, face_color='red')
    # viewer.add_image(target_heatmap_volume.cpu().numpy(), name='Target Heatmaps Volume', colormap='blue')
    # viewer.add_image(dataset.grandmodel_fbp[model_num].cpu().numpy(), name='Grandmodel FBP volume')
    # this_output_heatmaps_volume = resize(this_output_heatmaps_volume, (512, 512, 512), order=1,
    #                                     preserve_range=True, anti_aliasing=True)
    # viewer.add_image(this_output_heatmaps_volume, name='Output Heatmaps Volume', colormap='magenta')
    # viewer.add_image(dataset.grandmodel[model_num].cpu().numpy(), name='Grandmodel Volume',
    #                 colormap='gray')
    # napari.run()

    target_coordinates_df = target_coordinates_dict[model_num]
    # We take order z, y, x because that's how peak_local_max returns them as well
    target_coordinates = torch.tensor(target_coordinates_df[['Z', 'Y', 'X']].values)

    pred_coords = coordinates.float()
    tgt_coords = target_coordinates.float()
    num_preds = pred_coords.shape[0]
    num_targets = tgt_coords.shape[0]

    fp = 0
    tp = 0
    fn = 0

    num_preds = pred_coords.shape[0]
    num_targets = tgt_coords.shape[0]
    # Compute pairwise distances (num_preds, num_targets)
    dists = torch.cdist(pred_coords, tgt_coords, p=2).cpu().numpy()
    # Assign each prediction to the closest target
    pred_to_target = dists.argmin(axis=1)
    pred_to_target_dist = dists[np.arange(num_preds), pred_to_target]

    # Remove predictions that are too far from any target (count as extra)
    assigned_preds = {}
    extra_predictions = 0
    for pred_idx, (tgt_idx, dist) in enumerate(zip(pred_to_target, pred_to_target_dist)):
        if dist > missing_pred_threshold:
            extra_predictions += 1
            fp += 1
        else:
            assigned_preds.setdefault(tgt_idx, []).append((pred_idx, dist))

    missed_predictions = 0
    matched_dists = []
    for tgt_idx in range(num_targets):
        preds_for_target = assigned_preds.get(tgt_idx, [])
        if len(preds_for_target) == 0:
            missed_predictions += 1
            fn += 1
        elif len(preds_for_target) == 1:
            # One prediction for this target
            matched_dists.append(preds_for_target[0][1])
            tp += 1
        else:
            # Multiple predictions: pick the closest, others are extra
            preds_for_target.sort(key=lambda x: x[1])
            matched_dists.append(preds_for_target[0][1])
            tp += 1
            extra_predictions += len(preds_for_target) - 1
            fp += len(preds_for_target) - 1

    avg_pixels_off = np.mean(matched_dists) if matched_dists else 0.0

    avg_pixel_loss = avg_pixels_off
    avg_missed_predictions = missed_predictions
    avg_extra_predictions = extra_predictions
    correct_predictions = len(matched_dists)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    evaluation_dict = {"avg_pixel_loss": avg_pixel_loss, "avg_missed_predictions": avg_missed_predictions,
                       "avg_extra_predictions": avg_extra_predictions, "correct_predictions": correct_predictions,
                       "precision": precision, "recall": recall, "f1_score": f1_score, "num_preds": num_preds,
                       "num_targets": num_targets}

    return evaluation_dict
