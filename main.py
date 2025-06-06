import argparse
import torch
import os
import datetime
import time
import json

import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

# Local imports
from model import TopdownHeatmapSimpleHead
from utils import create_folder_if_missing
from evaluate import evaluate
from train import prepare_dataloaders, find_optimal_assignment_heatmaps, get_dataset, get_targets
from plotting import save_image_with_bounding_object, plot_loss_log, compare_heatmaps_with_ground_truth
from vit_model import get_vit_model, get_encoded_image


## Set the seed for reproducibility
#seed = 42
#random.seed(seed)
#torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    # Program Arguments
    parser.add_argument("--config", type=str, default="",
                        help="Path to the configuration file")
    parser.add_argument("--mode", type=str, help="Mode to run the program in: train, eval")
    parser.add_argument("--existing_result_folder", type=str, default="",
                        help="Path to existing result folder to load model from.")
    parser.add_argument("--existing_evaluation_folder", type=str, default="",
                        help="Name (not path) of existing evaluation folder within the specified experiment folder."
                             "Added so that you can reevaluate volumes or revisualize them without having to recompute"
                             "the entire volume.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    # Experiment Results
    parser.add_argument("--result_dir", type=str,
                        default=f'experiments/experiment_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
                        help="Directory to save results to")
    parser.add_argument("--result_dir_appended_name", type=str, default="",
                        help="Extra string to append to the end of the result directory")
    parser.add_argument("--use_train_dataset_for_evaluation", type=bool, default=False)

    # Training
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
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

    # Evaluation
    parser.add_argument('--prediction_threshold', type=float,
                        help='Threshold from which the maximum of a heatmap is considered to be a prediction')
    parser.add_argument('--neighborhood_size', type=int,
                        help='Determines the neighborhood size for predicting'
                             'local maxima on heatmap')
    parser.add_argument('--volume_evaluation', type=bool,
                        help='If True, the evaluation will be take a 3d volume as input, segment it into z slices'
                             'and then evaluate everything such that you get all 3d coordinates of the volume.')

    # Data and Model
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_patch_embeddings", type=int, default=196, help="Number of patch"
                                                                              "embeddings that the vit model generates,"
                                                                              "this is based on the patch size of the"
                                                                              "vit model")
    # Dataset general
    parser.add_argument("--dataset", type=str, help="Which dataset to use for running the program: dummy, shrec")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--vit_input_size", type=int, help="Size of image that is put through the vit",
                        default=224)
    # TODO: add checker for when num_particles is somehow less than the ground truth ones in the sub micrograph
    parser.add_argument("--num_particles", type=int, help="Number of particles that the model outputs as predictions")
    parser.add_argument("--particle_width", type=int)
    parser.add_argument("--particle_height", type=int)
    parser.add_argument("--particle_depth", type=int, help="Only relevenat for shrec dataset and others"
                                                           "that use 3d volumes to take slices from.")
    parser.add_argument("--add_noise", type=bool, help="Whether to add noise to the image or not")
    parser.add_argument("--noise", type=float, help="Level of noise to add to the dataset. Given in dB and"
                                                    "correspond to SNR that the noisy image should have compared to the"
                                                    "normal one")
    parser.add_argument("--gaussians_3d", type=bool, help="Whether to use 3d gaussians for the heatmaps or"
                                                          "not")
    parser.add_argument("--use_fbp", type=bool, help="Whether to use a simulated fbp of the volume or not")
    parser.add_argument("--fbp_num_projections", type=int, help="Number of projections for fbp simulation"
                                                                "on volume.")
    parser.add_argument("--fbp_min_angle", type=float, default=-torch.pi/3,
                        help="Minimum angle of fbp simulation given in radians")
    parser.add_argument("--fbp_max_angle", type=float, default=torch.pi/3,
                        help="Maximum angle of fbp simulation given in radians")

    # Dummy Dataset
    parser.add_argument("--dataset_size", type=int)

    # Shrec Dataset
    parser.add_argument("--shrec_sampling_points", type=int,
                        help="Number of points to use per z slice when generating sub micrographs for the shrec "
                             "dataset. The resulting number of sub micrographs will be "
                             "sampling_points^2 * num_z_slices")
    parser.add_argument("--shrec_z_slice_size", type=int, help="Size in pixels for z slices for the shrec "
                                                               "dataset")
    parser.add_argument("--shrec_model_number", type=int, default=1,
                        help="Which model of the shrec dataset to use, there are models from 0 to 9")
    parser.add_argument("--shrec_min_z", type=int, default=150, help="Minimum z to start taking z slices "
                                                                     "from for the shrec dataset")
    parser.add_argument("--shrec_max_z", type=int, default=360, help="Maximum z to start taking z slices "
                                                                     "from for the shrec dataset")
    # TODO: Add support for multiple particles maybe?
    parser.add_argument("--shrec_specific_particle", type=str, default=None,
                        help="If specified, the dataset will only create 3d gaussians for the specified particle.")

    # Outdated / Not used anymore
    # Matcher for crytransformer loss
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients for cryo transformer loss
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--quartile_threshold', type=float, default=0.0, help='Quartile threshold')  # TODO: Probably not needed anymore

    args = parser.parse_args()

    # Load arguments from configuration file if provided
    if args.config:
        print(f"Recognized configuration file: {args.config}")
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    args.result_dir = args.result_dir + args.result_dir_appended_name

    if args.existing_result_folder is not None and args.mode == "eval":
        args.result_dir = os.path.join('experiments', args.existing_result_folder)

    args.loss_log_path = os.path.join(args.result_dir, args.loss_log_file)

    if args.mode == "eval":
        print("Running in evaluation mode")
    elif args.mode == "train":
        print("Running in training mode")

    # Some validation here
    if args.use_fbp and not args.gaussians_3d:
        raise Exception("You can't set use_fbp to True but not gaussians_3d. use_fbp requires that you use 3d "
                        "gaussians.")
    if args.existing_evaluation_folder != "" and args.existing_result_folder == "":
        raise Exception("You specified an existing evaluation folder but not an experiment folder, please specify the "
                        "experiment folder in which the existing evaluation folder exists.")

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
    create_folder_if_missing(os.path.join(args.result_dir, 'training_examples'))

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

    # Dataset
    print("Creating Dataset...")
    dataset = get_dataset(args.dataset, args)

    # We only need to create the split file if were training, otherwise we read from it
    train_dataloader, test_dataloader = prepare_dataloaders(dataset=dataset, train_eval_split=args.train_eval_split,
                                                            batch_size=args.batch_size,
                                                            result_dir=args.result_dir,
                                                            split_file_name=args.split_file_name,
                                                            create_split_file=args.mode == "train",
                                                            use_train_dataset_for_evaluation=
                                                            args.use_train_dataset_for_evaluation)

    model = TopdownHeatmapSimpleHead(in_channels=args.latent_dim, out_channels=1)

    model.init_weights()
    model.to(args.device)

    mse_loss = torch.nn.MSELoss()

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model.train()

        last_checkpoint_time = time.time()
        # This is different from batch_index as this only counts how many batches have been done since the last avg_loss
        # calculation
        batch_counter = 0
        running_loss = 0.0

        # Save untrained checkpoint for debugging purposes
        torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_untrained.pth'))

        plotted = 0
        for epoch in range(args.epochs):
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')

            for batch_index, (micrographs, index) in enumerate(train_dataloader):
                batch_counter += 1

                target_heatmaps, targets = get_targets(args=args, dataset=dataset, index=index)

                if plotted < 5:  # TODO: move this into seperate function
                    save_image_with_bounding_object(micrographs[0].cpu(), targets[0]['boxes'].cpu()*args.vit_input_size,
                                                    "output_box",
                                                    {},
                                                    os.path.join(args.result_dir, 'training_examples'),
                                                    f"train_test_example_{plotted}")

                    compare_heatmaps_with_ground_truth(micrograph=micrographs[0].cpu(),
                                                       particle_locations=targets[0]['boxes'].cpu()*args.vit_input_size,
                                                       heatmaps=target_heatmaps[0].cpu(),
                                                       heatmaps_title="target heatmaps",
                                                       result_folder_name=f"ground_truth_vs_heatmaps_targets_{plotted}",
                                                       result_dir=os.path.join(args.result_dir, 'training_examples'))

                    vit_processed_micrographs = vit_image_processor(images=micrographs, return_tensors="pt",
                                                                    do_rescale=False)
                    # Output of this is between -1 and 1
                    vit_example_input = vit_processed_micrographs["pixel_values"][0].permute(1, 2, 0)
                    if vit_example_input.max() > 0:
                        vit_example_input = (vit_example_input - vit_example_input.min()) / (vit_example_input.max() -
                                                                                             vit_example_input.min())
                    if vit_example_input.max() == -1:  # In this case we got a fully black image
                        vit_example_input += 1
                    plt.imshow(vit_example_input)
                    plt.title(f"Micrograph after running through vit processor, min={vit_processed_micrographs['pixel_values'][0].min()}, max={vit_processed_micrographs['pixel_values'][0].max()}")
                    plt.savefig(os.path.join(args.result_dir, 'training_examples',
                                             f"vit_processed_micrograph_{plotted}.png"))
                    plt.close()
                    plotted += 1

                encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor)

                # 1: here because we don't need the class token
                latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
                # Right shape for model, we permute the hidden dimension to the second place
                # TODO: technically you could adjust the 14, 14 to be calculated but its unnecessary as long as you don't
                #  change the vit input size
                latent_micrographs = latent_micrographs.permute(0, 2, 1).reshape(args.batch_size, args.latent_dim,
                                                                                 14, 14)
                outputs = model(latent_micrographs)

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

                avg_loss = running_loss / batch_counter
                # Save running loss to log file
                with open(args.loss_log_path, 'a') as f:
                    f.write(f"{epoch},{batch_index},{avg_loss}\n")

                if time.time() - last_checkpoint_time > args.checkpoint_interval:
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
        if not os.path.exists(os.path.join(args.result_dir, "losses_plot.png")):
            plot_loss_log(args.loss_log_path, args.result_dir)

        checkpoint_path = os.path.join(args.result_dir, 'checkpoint_final.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        evaluate(args=args, criterion=mse_loss, vit_model=vit_model, vit_image_processor=vit_image_processor,
                 model=model, dataset=dataset, test_dataloader=test_dataloader, example_predictions=8)


if __name__ == "__main__":
    main()
