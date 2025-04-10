import torch
import json
import os

from torch.utils.data import DataLoader, random_split, Subset

# Local imports


def prepare_dataloaders(dataset, train_eval_split, batch_size, result_dir, split_file_name, create_split_file):
    """
    :param dataset: instance of torch dataset class
    :param train_eval_split: number between 0 and 1 determining how much should be training vs evaluation, 0.9 means
    90 percent of data is for training, 10 percent for evaluating
    :param batch_size: number determining batch size
    :param result_dir:
    :param split_file_name:
    :param create_split_file: creates one if set to true, otherwise it reads from an existing one
    :return: train_dataloader, test_dataloader
    """
    if not create_split_file:
        # Read split from file if one exists
        with open(os.path.join(result_dir, split_file_name), 'r') as f:
            split_indices = json.load(f)
        train_indices = split_indices['train']
        test_indices = split_indices['test']
    else:
        # Save split to file if we don't have one yet, so we can reproduce
        dataset_size = len(dataset)
        train_size = int(train_eval_split * dataset_size)
        indices = torch.randperm(dataset_size).tolist()
        train_indices, test_indices = indices[:train_size], indices[train_size:]

        with open(os.path.join(result_dir, split_file_name), 'w') as f:
            json.dump({'train': train_indices, 'test': test_indices}, f)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

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
