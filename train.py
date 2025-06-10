import torch
import json
import os

from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates


def prepare_dataloaders(dataset, train_eval_split, batch_size, result_dir, split_file_name, create_split_file,
                        use_train_dataset_for_evaluation):
    """
    :param dataset: instance of torch dataset class
    :param train_eval_split: number between 0 and 1 determining how much should be training vs evaluation, 0.9 means
    90 percent of data is for training, 10 percent for evaluating
    :param batch_size: number determining batch size
    :param result_dir:
    :param split_file_name:
    :param create_split_file: creates one if set to true, otherwise it reads from an existing one
    :param use_train_dataset_for_evaluation
    :return: train_dataloader, test_dataloader
    """
    if not create_split_file and os.path.exists(os.path.join(result_dir, split_file_name)):
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

    # We make our custom collate function here so that the class can handle variable length target coordinates
    def collate_fn(batch):
        micrographs, heatmaps, target_coords_list, debug_tuple = zip(*batch)
        micrographs = torch.stack(micrographs)
        heatmaps = torch.stack(heatmaps)
        targets_list = list(target_coords_list)

        return micrographs, heatmaps, targets_list, debug_tuple

    # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every
    # batch needs to be of size args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, drop_last=True, collate_fn=collate_fn)

    if use_train_dataset_for_evaluation:
        test_dataloader = DataLoader(train_dataset, batch_size=1)

    return train_dataloader, test_dataloader


def find_optimal_assignment_heatmaps(model_heatmaps, target_heatmaps, loss_fn):
    """
    Finds the optimal assignment of model outputs to target outputs to minimize the loss.

    Args:
        model_heatmaps (torch.Tensor): Model outputs of shape (batch_size, num_predictions, 112, 112).
        target_heatmaps (torch.Tensor): Target heatmaps of shape (batch_size, num_predictions, 112, 112).
        loss_fn (callable): Loss function to compute the cost between heatmaps.

    Returns:
        list: A list of optimal assignments for each batch.
    """
    batch_size, num_predictions, _, _ = model_heatmaps.shape
    assignments = []

    for b in range(batch_size):
        # Compute the cost matrix (num_predictions x num_predictions)
        cost_matrix = torch.zeros((num_predictions, num_predictions), device=model_heatmaps.device)
        for i in range(num_predictions):
            for j in range(num_predictions):
                cost_matrix[i, j] = loss_fn(model_heatmaps[b, i], target_heatmaps[b, j])

        # Convert the cost matrix to numpy for the Hungarian algorithm
        cost_matrix_np = cost_matrix.detach().cpu().numpy()

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        # Store the assignment (row_ind corresponds to model outputs, col_ind to target heatmaps)
        assignments.append((row_ind, col_ind))

    return assignments
