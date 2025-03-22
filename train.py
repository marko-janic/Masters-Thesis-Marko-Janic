import torch

from torch.utils.data import DataLoader, random_split

# Local imports
from dataset import get_particle_locations_from_coordinates


def prepare_dataloaders(dataset, train_eval_split, batch_size):
    """
    :param dataset: instance of torch dataset class
    :param train_eval_split: number between 0 and 1 determining how much should be training vs evaluation, 0.9 means
    90 percent of data is for training, 10 percent for evaluating
    :param batch_size: number determining batch size
    :return: train_dataloader, test_dataloader
    """
    train_size = int(train_eval_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every
    # batch needs to be of size args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return train_dataloader, test_dataloader


def compute_losses(args, index, dataset, model, vit_model, micrographs, criterion):
    # TODO: I need to remove this or refactor this somehow
    model.train()
    criterion.train()

    targets = None
    if args.dataset == "shrec":
        targets = prepare_targets_for_loss(args, index, dataset)
    elif args.dataset == "dummy":
        targets = []
        for target_index in index:
            targets.append(dataset.targets[target_index])
    if targets is None:
        raise Exception(f"Targets was not assigned. Either your args.dataset (which is {args.dataset}) is not set or "
                        f"the dataset you set is wrong.")

    latent_micrographs = vit_model(micrographs)
    predictions = model(latent_micrographs)
    outputs = prepare_outputs_for_loss(predictions)

    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    #optimizer.zero_grad()
    #losses.backward()
    #optimizer.step()

    return losses, outputs, targets


# TODO move to shrec class
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


# TODO move to shrec class
def prepare_outputs_for_loss(predictions):
    # Again we do it to fit the cryo transformer loss
    predictions_classes = predictions[:, :, 4:]
    predictions_coordinates = predictions[:, :, :4]
    outputs = {
        "pred_logits": predictions_classes,
        "pred_boxes": predictions_coordinates
    }

    return outputs
