import argparse
import torch
import os
import datetime
import time
import json

import torch.optim as optim

from tqdm import tqdm

# Local imports
from model import TopdownHeatmapSimpleHead
from util.utils import create_folder_if_missing
from dataset import DummyDataset
from loss import build
from evaluate import evaluate
from train import prepare_dataloaders, create_heatmaps_from_targets, find_optimal_assignment_heatmaps
from plotting import save_image_with_bounding_object, plot_loss_log
from vit_model import get_vit_model, get_encoded_image


# Set the seed for reproducibility
# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    # Program Arguments
    parser.add_argument("--config", type=str, default="run_configs/dummy_dataset_evaluation.json",
                        help="Path to the configuration file")
    parser.add_argument("--dataset", type=str, help="Which dataset to use for running the program: dummy")
    parser.add_argument("--mode", type=str, help="Mode to run the program in: train, eval")
    parser.add_argument("--existing_result_folder", type=str, default="",
                        help="Path to existing result folder to load model from.")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_size", type=int)
    # TODO: add checker for when num_particles is somehow less than the ground truth ones in the sub micrograph
    parser.add_argument("--num_particles", type=int, help="Number of particles that the model outputs as predictions")
    parser.add_argument("--particle_width", type=int, default=80)
    parser.add_argument("--particle_height", type=int, default=80)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    # Experiment Results
    parser.add_argument("--result_dir", type=str,
                        default=f'experiments/experiment_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
                        help="Directory to save results to")
    parser.add_argument("--result_dir_appended_name", type=str, default="",
                        help="Extra string to append to the end of the result directory")
    parser.add_argument("--use_train_dataset_for_evaluation", type=bool, default=False)

    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--checkpoint_interval", type=int,
                        help="Save model checkpoint every checkpoint_interval seconds")

    parser.add_argument("--loss_log_file", type=str, default="loss_log.txt",
                        help="File to save running loss for each epoch")
    parser.add_argument("--train_eval_split", type=float, default=0.9,
                        help="Ratio of training to evaluation split. 0.9 means that 90% of the data is used for "
                             "training and 10% for evaluation")
    parser.add_argument("--split_file_name", type=str, default="dataset_split_indices.json",
                        help="File with dataset split indices. Used to get the same train test split after program has"
                             "already been run")

    # Data and Model
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_patch_embeddings", type=int, default=196, help="Number of patch"
                                                                              "embeddings that the vit model generates,"
                                                                              "this is based on the patch size of the"
                                                                              "vit model")

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Evaluation
    parser.add_argument('--quartile_threshold', type=float, default=0.0, help='Quartile threshold')  # TODO: Needs refactoring

    args = parser.parse_args()

    # Load arguments from configuration file if provided
    if args.config:
        print(f"Recognized configuration file: {args.config}")
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    args.result_dir = args.result_dir + args.result_dir_appended_name
    args.loss_log_path = ""
    if args.mode != "eval":
        args.loss_log_path = os.path.join(args.result_dir, args.loss_log_file)

    if args.existing_result_folder is not None and args.mode == "eval":
        args.result_dir = os.path.join('experiments', args.existing_result_folder)

    if args.mode == "eval":
        print("Running in evaluation mode")
    elif args.mode == "train":
        print("Running in training mode")

    return args


def create_folders_and_initiate_files(args):
    """
    Creates necessary folders and initializes files for logging.

    :param args: Requires the following arguments in args:
        - result_dir: Directory to save results.
        - existing_result_folder: Path to an existing result folder (used in evaluation mode).
        - mode: Mode of operation
        - loss_log_path: Path to the loss log file.
    :return: None
    """
    create_folder_if_missing(args.result_dir)
    create_folder_if_missing(os.path.join(args.result_dir, 'checkpoints'))

    # Save Training information into file
    if args.mode != "eval":
        with open(os.path.join(args.result_dir, 'arguments.txt'), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

    # Initialize loss log file
    if args.mode != "eval":
        with open(args.loss_log_path, 'w') as f:
            f.write("epoch,batch,average_loss\n")


def main():
    args = get_args()
    create_folders_and_initiate_files(args)

    # ViT model
    vit_model, vit_image_processor = get_vit_model()

    # Training
    dataset = DummyDataset(dataset_size=args.dataset_size, dataset_path=args.dataset_path,
                           particle_width=args.particle_width, particle_height=args.particle_height)

    # We only need to create the split file if were training, otherwise we read from it
    train_dataloader, test_dataloader = prepare_dataloaders(dataset=dataset, train_eval_split=args.train_eval_split,
                                                            batch_size=args.batch_size,
                                                            result_dir=args.result_dir,
                                                            split_file_name=args.split_file_name,
                                                            create_split_file=args.mode == "train",
                                                            use_train_dataset_for_evaluation=
                                                            args.use_train_dataset_for_evaluation)

    model = TopdownHeatmapSimpleHead(in_channels=args.latent_dim,
                                     out_channels=args.num_particles)
    model.init_weights()
    model.to(args.device)

    criterion, postprocessors = build(args)
    mse_loss = torch.nn.MSELoss()

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model.train()
        criterion.train()

        last_checkpoint_time = time.time()
        # This is different from batch_index as this only counts how many batches have been done since the last avg_loss
        # calculation
        batch_counter = 0
        running_loss = 0.0

        # Save untrained checkpoint for debugging purposes
        torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_untrained.pth'))

        plotted = False
        for epoch in range(args.epochs):
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

            for batch_index, (micrographs, index) in enumerate(train_dataloader):
                batch_counter += 1

                targets = dataset.get_targets_from_target_indexes(index, args.device)
                target_heatmaps = create_heatmaps_from_targets(targets, num_predictions=args.num_particles,
                                                               device=args.device)

                if not plotted:
                    save_image_with_bounding_object(micrographs[0].cpu()/255, targets[0]['boxes'].cpu()*224, "output_box",  # TODO: This 224 is hacky, fix it
                                                    {}, args.result_dir, "train_test_example")
                    plotted = True

                encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor)

                latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
                outputs = model(latent_micrographs.reshape((args.batch_size, args.latent_dim, 14, 14)))  # TODO: don't hardcode this 14

                assignments = find_optimal_assignment_heatmaps(outputs["heatmaps"], target_heatmaps, mse_loss)

                reordered_target_heatmaps = torch.zeros_like(target_heatmaps)
                for batch_idx, (row_ind, col_ind) in enumerate(assignments):
                    reordered_target_heatmaps[batch_idx] = target_heatmaps[batch_idx, col_ind]

                losses = mse_loss(outputs["heatmaps"], reordered_target_heatmaps)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                epoch_bar.set_postfix(loss=losses.item())
                epoch_bar.update(1)

                if time.time() - last_checkpoint_time > args.checkpoint_interval:
                    avg_loss = running_loss / batch_counter

                    # Save running loss to log file
                    with open(args.loss_log_path, 'a') as f:
                        f.write(f"{epoch},{batch_index},{avg_loss}\n")

                    # Save checkpoint
                    torch.save(model.state_dict(), os.path.join(args.result_dir,
                                                                f'checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}.pth'))

                    last_checkpoint_time = time.time()
                    batch_counter = 0
                    running_loss = 0.0

            epoch_bar.close()

        # Save final checkpoint
        torch.save(model.state_dict(), os.path.join(args.result_dir, 'checkpoint_final.pth'))
        # Plot the loss log after training
        plot_loss_log(args.loss_log_path, args.result_dir)

    if args.mode == "eval":
        checkpoint_path = os.path.join(args.result_dir, 'checkpoint_final.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        evaluate(args=args, criterion=mse_loss, vit_model=vit_model, vit_image_processor=vit_image_processor,
                 model=model, dataset=dataset, test_dataloader=test_dataloader, example_predictions=8)


if __name__ == "__main__":
    main()
