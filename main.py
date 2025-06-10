import argparse
import torch
import os
import datetime
import time
import json
import random

import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm

from dataset import ShrecDataset
# Local imports
from model import TopdownHeatmapSimpleHead
from utils import create_folder_if_missing
from evaluate import evaluate
from train import prepare_dataloaders
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
    parser.add_argument("--finetune_vit", type=bool, default=False,
                        help="Determines whether to fintune a vit during training or not. If set to true while "
                             "evaluating the program will try to load an already finetuned vit from the experiment "
                             "folder")

    # Evaluation
    parser.add_argument('--prediction_threshold', type=float,
                        help='Threshold from which the maximum of a heatmap is considered to be a prediction')
    parser.add_argument('--neighborhood_size', type=int,
                        help='Determines the neighborhood size for predicting'
                             'local maxima on heatmap')
    parser.add_argument('--volume_evaluation', type=bool,
                        help='If True, the evaluation will be take a 3d volume as input, segment it into z slices'
                             'and then evaluate everything such that you get all 3d coordinates of the volume.')
    parser.add_argument("--missing_pred_threshold", type=int,
                        help="Predictions that are further away than this parameter (in pixels) from a valid target "
                             "will be counted as missed predictions")

    # Data and Model
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_patch_embeddings", type=int, default=196, help="Number of patch"
                                                                              "embeddings that the vit model generates,"
                                                                              "this is based on the patch size of the"
                                                                              "vit model")
    parser.add_argument("--model_deconv_filters", type=tuple, help="Tuple determining the 3 layer deconv"
                                                                   "filter size for the model. For example"
                                                                   "[256, 256, 256]")
    parser.add_argument("--dropout_prob", type=float, help="Determines the dropout probability for"
                                                           "Dropout layers in the model. For example 0.1")
    parser.add_argument("--heatmap_size", type=int, help="Size of the heatmaps that are fed into the model"
                                                         "by default its just 112 which means that the heatmaps"
                                                         "are of shape 112 x 112")

    # Dataset general
    parser.add_argument("--dataset", type=str, help="Which dataset to use for running the program: shrec")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--vit_input_size", type=int, help="Size of image that is put through the vit",
                        default=224)
    parser.add_argument("--particle_width", type=int)
    parser.add_argument("--particle_height", type=int)
    parser.add_argument("--particle_depth", type=int, help="Only relevenat for shrec dataset and others"
                                                           "that use 3d volumes to take slices from.")
    parser.add_argument("--add_noise", type=bool, help="Whether to add noise to the image or not")
    parser.add_argument("--noise", type=float, help="Level of noise to add to the dataset. Given in dB and"
                                                    "correspond to SNR that the noisy image should have compared to the"
                                                    "normal one")
    parser.add_argument("--use_fbp", type=bool, help="Whether to use a simulated fbp of the volume or not")
    parser.add_argument("--fbp_num_projections", type=int, help="Number of projections for fbp simulation"
                                                                "on volume.")
    parser.add_argument("--fbp_min_angle", type=float, default=-torch.pi/3,
                        help="Minimum angle of fbp simulation given in radians")
    parser.add_argument("--fbp_max_angle", type=float, default=torch.pi/3,
                        help="Maximum angle of fbp simulation given in radians")

    # Shrec Dataset
    parser.add_argument("--shrec_sampling_points", type=int,
                        help="Number of points to use per z slice when generating sub micrographs for the shrec "
                             "dataset. The resulting number of sub micrographs will be "
                             "sampling_points^2 * num_z_slices")
    parser.add_argument("--shrec_z_slice_size", type=int, help="Size in pixels for z slices for the shrec "
                                                               "dataset")
    parser.add_argument("--shrec_model_number", type=int, default=[1],
                        help="Which models of the shrec dataset to use, there are models from 0 to 9. "
                             "Has to be specified as a list and these are the models that will be used during training."
                             "For evaluation it will only take the first element in the list, irrespective of what"
                             "you specify after in the list")
    parser.add_argument("--shrec_min_z", type=int, default=150, help="Minimum z to start taking z slices "
                                                                     "from for the shrec dataset")
    parser.add_argument("--shrec_max_z", type=int, default=360, help="Maximum z to start taking z slices "
                                                                     "from for the shrec dataset")
    # TODO: Add support for multiple particles maybe?
    parser.add_argument("--shrec_specific_particle", type=str, default=None,
                        help="If specified, the dataset will only create 3d gaussians for the specified particle.")

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
    vit_model.to(args.device)
    if args.mode == "eval" and args.finetune_vit:
        print("Loading finetuned ViT model")
        vit_model.load_state_dict(torch.load(os.path.join(args.result_dir, 'vit_checkpoint_final.pth'),
                                             map_location=args.device))

    # Freeze all parameters (finetuning vit)
    for param in vit_model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 trasnformer blocks (finetuning vit)
    if args.finetune_vit:
        for block in vit_model.encoder.layer[-2:]:
            for param in block.parameters():
                param.requires_grad = True

    # Dataset
    dataset = ShrecDataset(sampling_points=args.shrec_sampling_points, z_slice_size=args.shrec_z_slice_size,
                           model_number=args.shrec_model_number, min_z=args.shrec_min_z, max_z=args.shrec_max_z,
                           particle_height=args.particle_height, particle_width=args.particle_width,
                           particle_depth=args.particle_depth, noise=args.noise, add_noise=args.add_noise,
                           device=args.device, use_fbp=args.use_fbp, fbp_min_angle=args.fbp_min_angle,
                           fbp_max_angle=args.fbp_max_angle, fbp_num_projections=args.fbp_num_projections,
                           shrec_specific_particle=args.shrec_specific_particle, heatmap_size=args.heatmap_size)

    # We only need to create the split file if were training, otherwise we read from it
    train_dataloader, test_dataloader = prepare_dataloaders(dataset=dataset, train_eval_split=args.train_eval_split,
                                                            batch_size=args.batch_size,
                                                            result_dir=args.result_dir,
                                                            split_file_name=args.split_file_name,
                                                            create_split_file=args.mode == "train",
                                                            use_train_dataset_for_evaluation=
                                                            args.use_train_dataset_for_evaluation)

    model = TopdownHeatmapSimpleHead(in_channels=args.latent_dim, out_channels=1,
                                     num_deconv_filters=tuple(args.model_deconv_filters),
                                     dropout_prob=args.dropout_prob)
    model.init_weights()
    model.to(args.device)

    mse_loss = torch.nn.MSELoss()

    if args.mode == "train":
        if args.finetune_vit:
            optimizer = optim.Adam(
                list(model.parameters()) + list(vit_model.parameters()),
                lr=args.learning_rate)
            model.train()
            vit_model.train()
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            model.train()
            vit_model.eval()

        last_checkpoint_time = time.time()

        # Save untrained checkpoint for debugging purposes
        torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_untrained.pth'))

        plotted = 0
        for epoch in range(args.epochs):
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')
            # This is different from batch_index as this only counts how many batches have been done since the last
            # avg_loss calculation
            batch_counter = 0
            running_loss = 0.0

            for batch_index, (micrographs, target_heatmaps, target_coordinates_list, debug_tuple) \
                    in enumerate(train_dataloader):
                batch_counter += 1

                if plotted < 20:  # TODO: move this into seperate function
                    save_image_with_bounding_object(
                        micrographs[0].cpu(), target_coordinates_list[0].cpu()*args.vit_input_size,
                        "circle", {"circle_radius": 6}, os.path.join(args.result_dir, 'training_examples'),
                        f"train_test_example_{plotted}_coords_xyz_{debug_tuple[0][0]}_"
                        f"orientation_{debug_tuple[0][1]}_model_{debug_tuple[0][2]}")

                    plt.close()
                    plt.imshow(micrographs[0].cpu().permute(1, 2, 0))
                    target_heatmap_resized = F.interpolate(target_heatmaps[0].unsqueeze(0), size=(224, 224),
                                                           mode='bilinear', align_corners=False).squeeze(0)
                    plt.imshow(target_heatmap_resized[0].cpu(), cmap='jet', alpha=0.25)
                    plt.axis('off')
                    plt.savefig(os.path.join(args.result_dir, 'training_examples',
                                             f"overlayed_gt_heatmaps_{plotted}.png"))
                    plt.close()

                    compare_heatmaps_with_ground_truth(
                        micrograph=micrographs[0].cpu(),
                        particle_locations=target_coordinates_list[0].cpu()*args.vit_input_size,
                        heatmaps=target_heatmaps[0].cpu(),
                        heatmaps_title=f"target heatmap",
                        result_file_name=f"ground_truth_vs_target_heatmap_{plotted}.png",
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
                    plt.title(f"Micrograph after running through vit processor, "
                              f"min={vit_processed_micrographs['pixel_values'][0].min()}, "
                              f"max={vit_processed_micrographs['pixel_values'][0].max()}")
                    plt.savefig(os.path.join(args.result_dir, 'training_examples',
                                             f"vit_processed_micrograph_{plotted}.png"))
                    plt.close()
                    plotted += 1

                encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor, device=args.device)

                # 1: here because we don't need the class token
                latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
                # Right shape for model, we permute the hidden dimension to the second place
                # TODO: technically you could adjust the 14, 14 to be calculated but its unnecessary as long as you don't
                #  change the vit input size
                latent_micrographs = latent_micrographs.permute(0, 2, 1)
                latent_micrographs = latent_micrographs.reshape(latent_micrographs.size(0), latent_micrographs.size(1),
                                                                14, 14)
                outputs = model(latent_micrographs)

                losses = mse_loss(outputs["heatmaps"], target_heatmaps)

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
                    torch.save(model.state_dict(), os.path.join(
                        args.result_dir, f'checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}.pth'))

                    if args.finetune_vit:
                        torch.save(vit_model.state_dict(), os.path.join(
                            args.result_dir, f"checkpoints/vit_checkpoint_epoch{epoch}_batch_{batch_index}.pth"))

                    last_checkpoint_time = time.time()

            epoch_bar.close()

        # Save final checkpoint
        torch.save(model.state_dict(), os.path.join(args.result_dir, f"checkpoint_final.pth"))
        if args.finetune_vit:
            torch.save(vit_model.state_dict(), os.path.join(args.result_dir, f"vit_checkpoint_final.pth"))

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

        with torch.no_grad():
            evaluate(args=args, criterion=mse_loss, vit_model=vit_model, vit_image_processor=vit_image_processor,
                     model=model, dataset=dataset, test_dataloader=test_dataloader, example_predictions=20)


if __name__ == "__main__":
    main()
