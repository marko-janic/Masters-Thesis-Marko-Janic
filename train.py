import torch

# Local imports
from dataset import get_particle_locations_from_coordinates


def prepare_targets_for_loss(args, coordinate_tl_list, dataset):
    """
    This is only for the shrec dataset
    :param coordinate_tl_list:
    :param dataset:
    :param num_particles:
    :return:
    """
    targets = []
    for index, coordinate_tl in enumerate(coordinate_tl_list):
        particle_locations = get_particle_locations_from_coordinates(coordinate_tl,
                                                                     dataset.sub_micrograph_size,
                                                                     dataset.particle_locations)
        # We do this so that it fits into the loss function given by cryo transformer
        boxes = torch.tensor(particle_locations[['X', 'Y']].values) / dataset.sub_micrograph_size
        width_height_columns = torch.ones((boxes.shape[0], 2)) * 0.001  # TODO: add this as an argument

        boxes = torch.cat((boxes, width_height_columns), dim=1)
        labels = torch.zeros(boxes.shape[0])
        orig_size = torch.tensor([dataset.sub_micrograph_size, dataset.sub_micrograph_size])

        ## Pad boxes to 500 entries (padded with 0, 0 for coordinates and 0.01 for other values)
        #pad_size = args.num_particles - boxes.shape[0]
        #if pad_size > 0:
        #    padding = torch.zeros(pad_size, 4)  # Pad with 0, 0, 0, 0 (4 elements for each box)
        #    boxes = torch.cat((boxes, padding), dim=0)
        ## Now, pad 'labels' to a fixed size of 500 with 0s
        #label_pad_size = args.num_particles - labels.shape[0]
        #if label_pad_size > 0:
        #    label_padding = torch.zeros(label_pad_size)  # Padding for labels (all 0s)
        #    labels = torch.cat((labels, label_padding), dim=0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": orig_size,
            "image_id": index,
        }
        targets.append(target)

    return targets


def prepare_outputs_for_loss(predictions):
    # Again we do it to fit the cryo transformer loss
    predictions_classes = predictions[:, :, 4:]
    predictions_coordinates = predictions[:, :, :4]
    outputs = {
        "pred_logits": predictions_classes,
        "pred_boxes": predictions_coordinates
    }

    return outputs


def train_model():
    pass
