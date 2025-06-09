import torch
import json
import os

from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

# Local imports
from dataset import DummyDataset, ShrecDataset, get_particle_locations_from_coordinates


def get_dataset(dataset_name, args):
    """
    Gives you back the dataset based on the specified dataset
    :param dataset_name:
    :param args: Required args are:
        noise,
        add_noise,
        gaussians_3d,
        device,
        use_fbp,
    :return:
    """
    if dataset_name == "dummy":
        return DummyDataset(dataset_size=args.dataset_size, dataset_path=args.dataset_path,
                            particle_width=args.particle_width, particle_height=args.particle_height)
    elif dataset_name == "shrec":
        return ShrecDataset(sampling_points=args.shrec_sampling_points, z_slice_size=args.shrec_z_slice_size,
                            model_number=args.shrec_model_number, min_z=args.shrec_min_z, max_z=args.shrec_max_z,
                            particle_height=args.particle_height, particle_width=args.particle_width,
                            particle_depth=args.particle_depth, noise=args.noise,
                            add_noise=args.add_noise, gaussians_3d=args.gaussians_3d, device=args.device,
                            use_fbp=args.use_fbp, fbp_min_angle=args.fbp_min_angle, fbp_max_angle=args.fbp_max_angle,
                            fbp_num_projections=args.fbp_num_projections,
                            shrec_specific_particle=args.shrec_specific_particle)
    else:
        raise Exception(f"The dataset {dataset_name}, is not supported.")


def get_targets(args, dataset, index, orientation):
    """
    Gives you back targets depending on your dataset
    :param args: Needs (See the args for docs):
        dataset,
        device,
        num_particles,
    :param dataset: torch dataset class
    :param index: Depends on the dataset:
        For dummy: just an index to use to call get_targets_from_target_indexes
        For shrec: torch.Tensor[batch_size, 3] representing the top left coordinates of all the micrographs in the
            batch. Again really just used to call its function get_target_heatmaps_from_3d_gaussians
    :param orientation: List of orientations of sub micrographs
    :return: Tuple(torch.Tensor[batch_size, num_predictions, heatmap_size, heatmap_size], list).
    The list is of length batch_size with dicts which under the "boxes" key have a torch.Tensor[4], representing
    particle locations.
    """
    if args.dataset == "dummy":
        targets = dataset.get_targets_from_target_indexes(index, args.device)
        target_heatmaps = create_heatmaps_from_targets(targets, num_predictions=args.num_particles,
                                                       device=args.device)
        return target_heatmaps, targets
    elif args.dataset == "shrec":
        targets = []
        for i in range(len(index)):
            selected_particles = get_particle_locations_from_coordinates(coordinates_tl=index[i],
                                                                         sub_micrograph_size=dataset.sub_micrograph_size,
                                                                         particle_locations=dataset.particle_locations,
                                                                         z_slice_size=dataset.z_slice_size,
                                                                         particle_width=dataset.particle_width,
                                                                         particle_height=dataset.particle_height,
                                                                         particle_depth=dataset.particle_depth,
                                                                         shrec_specific_particle=args.shrec_specific_particle,
                                                                         orientation=orientation[i])
            # We flip it here because of the coordinates being weird
            selected_particles["boxes"][:, 1] = dataset.sub_micrograph_size - selected_particles["boxes"][:, 1]
            selected_particles["boxes"] /= 224
            targets.append(selected_particles)

        if args.gaussians_3d:
            target_heatmaps = dataset.get_target_heatmaps_from_3d_gaussians(index, batch_size=args.batch_size,
                                                                            orientations=orientation)
        else:
            target_heatmaps = create_heatmaps_from_targets(targets, num_predictions=args.num_particles,
                                                           device=args.device)

        return target_heatmaps, targets
    else:
        raise Exception(f"The dataset {dataset}, is not supported.")


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

    # Important to set drop_last=True otherwise certain bath_size + dataset combinations don't work since every
    # batch needs to be of size args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    if use_train_dataset_for_evaluation:
        test_dataloader = DataLoader(train_dataset, batch_size=1)

    return train_dataloader, test_dataloader


def create_heatmaps_from_targets(data_list, num_predictions, device):
    """
    Create heatmaps for particles in a batch of images using PyTorch's MultivariateNormal.

    Args:
        data_list (list of dict): A list of dictionaries, each containing a key "boxes" with a torch tensor of size
        (num_particles x 4).
        num_predictions (int): Number of predictions (heatmaps) to generate for each image.
        device:

    Returns:
        torch.Tensor: A tensor of size (batch_size x num_predictions x 112 x 112).
    """
    batch_size = len(data_list)
    heatmap_size = 112  # TODO: don't hardcode this

    output = torch.full((batch_size, num_predictions, heatmap_size, heatmap_size), 0.0)

    for batch_idx, data in enumerate(data_list):
        boxes = data["boxes"]  # Tensor of size (num_particles x 4)
        num_particles = boxes.size(0)

        for particle_idx in range(min(num_particles, num_predictions)):
            center_x, center_y, particle_width, particle_height = boxes[particle_idx]
            center_x = center_x.item() * heatmap_size
            center_y = center_y.item() * heatmap_size
            sigma_x = (particle_width.item() * heatmap_size) / 3  # Approximation for Gaussian spread
            sigma_y = (particle_height.item() * heatmap_size) / 3

            # Define the Gaussian distribution
            mean = torch.tensor([center_x, center_y])
            covariance_matrix = torch.tensor([[sigma_x**2, 0], [0, sigma_y**2]])
            gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix)

            # Create a grid of coordinates
            y, x = torch.meshgrid(torch.arange(heatmap_size), torch.arange(heatmap_size), indexing="ij")
            # Invert the y-axis to match the image coordinate system
            y = heatmap_size - 1 - y

            grid = torch.stack([x.flatten(), y.flatten()], dim=-1)

            # Evaluate the Gaussian on the grid
            heatmap = gaussian.log_prob(grid).exp().reshape(heatmap_size, heatmap_size)

            # Normalize the heatmap so the value at the mean is exactly 1
            peak_value = 1 / (2 * torch.pi * sigma_x * sigma_y)
            heatmap /= peak_value

            # Add the Gaussian to the heatmap
            output[batch_idx, particle_idx] = heatmap

    output = output.max(dim=1, keepdim=True)[0]  # at [1] it would be the indices but we don't care about those

    return output.to(device)


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
