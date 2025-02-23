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
from train import prepare_targets_for_loss, prepare_outputs_for_loss


def evaluate(args, model, vit_model, dataset, test_dataloader, criterion, example_predictions, postprocessors):
    result_dir = os.path.join(args.result_dir, f'evaluation_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    create_folder_if_missing(result_dir)

    model.eval()
    criterion.eval()
    running_loss = 0.0

    example_counter = 0
    with torch.no_grad():
        for sub_micrographs, coordinate_tl_list in tqdm(test_dataloader, desc="Evaluating"):
            targets = prepare_targets_for_loss(args, coordinate_tl_list, dataset)

            latent_sub_micrographs = vit_model(sub_micrographs)
            predictions = model(latent_sub_micrographs)

            outputs = prepare_outputs_for_loss(predictions)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                target_sizes = torch.tensor([(224, 224)] * 1)
                results = postprocessors['bbox'](outputs, target_sizes)
                # Extract coordinates from results and convert to DataFrame
                pred_coords = results[0]['boxes'][:, :2].cpu().numpy()
                pred_coords_df = pd.DataFrame(pred_coords, columns=['X', 'Y'])

                ## Transform predictions coordinates tensor into a dataframe
                #pred_coords = outputs["pred_boxes"][0, :, :2].cpu().numpy()
                #pred_coords = transform_coords_to_pixel_coords(dataset.sub_micrograph_size, dataset.sub_micrograph_size,
                #                                               pred_coords)
                #pred_probs = outputs["pred_logits"][0, :, 0].cpu().numpy()
                ## TODO: again dont make this a hard coded value
                ##pred_coords_df = pd.DataFrame(pred_coords[pred_probs > 0.5], columns=['X', 'Y'])
                #pred_coords_df = pd.DataFrame(pred_coords, columns=['X', 'Y'])

                ground_truth = transform_coords_to_pixel_coords(dataset.sub_micrograph_size, dataset.sub_micrograph_size, targets[0]['boxes'][:, :2])
                ground_truth_df = pd.DataFrame(ground_truth, columns=['X', 'Y'])

                compare_predictions_with_ground_truth(
                    image_tensor=sub_micrographs[0].cpu(),
                    ground_truth=ground_truth_df,
                    predictions=pred_coords_df,
                    circle_radius=5,
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
