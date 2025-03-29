import os
import datetime
import torch

import numpy as np

from tqdm import tqdm

# Local imports
from util.utils import create_folder_if_missing, transform_coords_to_pixel_coords
from plotting import compare_predictions_with_ground_truth
from vit_model import get_encoded_image
from train import prepare_outputs_for_loss


def evaluate(args, model, vit_model, vit_image_processor, dataset, test_dataloader, criterion, example_predictions,
             postprocessors):
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

            encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor)
            if args.only_use_class_token:
                latent_micrographs = encoded_image['pooler_output'].to(args.device)
            else:
                latent_micrographs = encoded_image['last_hidden_state'].to(args.device)
            predictions = model(latent_micrographs)
            outputs = prepare_outputs_for_loss(predictions)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            running_loss += losses.item()

            if example_counter < example_predictions:  # TODO: needs a lot of refactoring
                target_sizes = torch.tensor([(224, 224)] * 1)
                results = postprocessors['bbox'](outputs, target_sizes)

                # This weird stuff is how cryotransformers does the predicting
                probas2 = outputs['pred_logits'].sigmoid()

                # topk_values, topk_indexes = torch.topk(probas2.view(outputs["pred_logits"].shape[0], -1), args.num_particles, dim=1)  # extreme important mention num queries
                # scores = topk_values
                # keep = scores[0] < np.quantile(scores, args.quartile_threshold)
                # scores = scores[0, keep]

                keep = probas2[0, :, 0] > probas2[0, :, 1]
                pred_coords = outputs['pred_boxes'][0, keep, :]
                pred_coords = transform_coords_to_pixel_coords(224, 224, pred_coords.cpu().numpy())
                ground_truth = transform_coords_to_pixel_coords(224, 224, targets[0]['boxes'][:, :4])

                compare_predictions_with_ground_truth(
                    image_tensor=micrographs[0].cpu(),
                    ground_truth=ground_truth,
                    predictions=pred_coords,
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
