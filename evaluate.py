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
from util.utils import create_folder_if_missing
from plotting import compare_predictions_with_ground_truth


def evaluate(model, vit_model, dataset, test_dataloader, criterion, experiment_dir, example_predictions, device):
    result_dir = os.path.join(experiment_dir, f'evaluation_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    create_folder_if_missing(result_dir)

    model.eval()
    criterion.eval()
    running_loss = 0.0

    example_counter = 0
    with torch.no_grad():
        for sub_micrographs, coordinate_tl_list in tqdm(test_dataloader, desc="Evaluating"):
            targets = []
            for index, coordinate_tl in enumerate(coordinate_tl_list):
                particle_locations = get_particle_locations_from_coordinates(coordinate_tl,
                                                                             dataset.sub_micrograph_size,
                                                                             dataset.particle_locations)
                boxes = torch.tensor(particle_locations[['X', 'Y']].values)
                zero_columns = torch.ones((boxes.shape[0], 2)) * 0.01
                boxes = torch.cat((boxes, zero_columns), dim=1)
                labels = torch.ones(boxes.shape[0])
                orig_size = torch.tensor([dataset.sub_micrograph_size, dataset.sub_micrograph_size])
                size = torch.tensor([dataset.sub_micrograph_size, dataset.sub_micrograph_size])

                pad_size = model.num_particles - boxes.shape[0]
                if pad_size > 0:
                    padding = torch.zeros(pad_size, 4)
                    boxes = torch.cat((boxes, padding), dim=0)

                label_pad_size = model.num_particles - labels.shape[0]
                if label_pad_size > 0:
                    label_padding = torch.zeros(label_pad_size)
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

            predictions_classes = predictions[:, :, 2:4]
            predictions_coordinates = predictions[:, :, :2]
            zeros = torch.ones(1, model.num_particles, 2) * 0.01
            predictions_coordinates = torch.cat((predictions_coordinates, zeros), dim=2)
            outputs = {
                "pred_logits": predictions_classes,
                "pred_boxes": predictions_coordinates
            }

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                # Transform predictions coordinates tensor into a dataframe
                ground_truth_df = pd.DataFrame(targets[0]['boxes'][:, :2].cpu().numpy(), columns=['X', 'Y'])
                pred_coords_df = pd.DataFrame(predictions_coordinates[0, :, :2].cpu().numpy(), columns=['X', 'Y'])
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
