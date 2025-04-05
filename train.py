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


# TODO move to corresponding dataset class
def prepare_outputs_for_loss(predictions):
    # Again we do it to fit the cryo transformer loss
    predictions_classes = predictions[:, :, 4:]
    predictions_coordinates = predictions[:, :, :4]
    outputs = {
        "pred_logits": predictions_classes,
        "pred_boxes": predictions_coordinates
    }

    return outputs
