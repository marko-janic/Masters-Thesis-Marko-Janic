import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from model import ParticlePicker
from loss import SetCriterion, build

def evaluate(model, vit_model, dataset, criterion, device):
    model.eval()
    criterion.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    running_loss = 0.0

    with torch.no_grad():
        for sub_micrographs, coordinate_tl_list in tqdm(dataloader, desc="Evaluating"):
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

    avg_loss = running_loss / len(dataloader)
    print(f"Average evaluation loss: {avg_loss}")
