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
from vit_model import get_encoded_image
from transformers import ViTModel, ViTImageProcessor, ViTConfig

## Set the seed for reproducibility
#seed = 42
#random.seed(seed)
#torch.manual_seed(seed)


def get_args():
    """
    This function specifies all adjustable arguments for this repository. This also includes a bit of sanity checking
    as well as loading arguments specified from a .json file.
    """
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
    parser.add_argument("--patience", type=int, help="Number of epochs before training stops if validation"
                                                     "loss doesn't improve in these epochs.")
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
    parser.add_argument("--num_vit_finetune_layers", type=int, default=False,
                        help="Determines how many of the last blocks of the vision transformer should be unfrozen and"
                             "then finetuned. Only works if the finetune_vit option is set to True")
    parser.add_argument("--shrec_validation_model_number", type=int, default=[9],
                        help="Shrec model volume to use for validating while training. Validation happens after"
                             "every epoch")
    parser.add_argument("--train_vit_from_scratch", type=bool, default=False,
                        help="If set to true then the program will train the vit from scratch without loading a"
                             "pretrained one")

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
    parser.add_argument("--find_optimal_parameters", type=bool,
                        help="If enabled a grid search will be performed to find optimal values for "
                             "prediction_threshold and neighborhood_size")

    # Data and Model
    parser.add_argument("--latent_dim", type=int, help="Dimensions of input to model")
    parser.add_argument("--num_patch_embeddings", type=int, default=196, help="Number of patch"
                                                                              "embeddings that the vit model generates,"
                                                                              "this is based on the patch size of the"
                                                                              "vit model")
    parser.add_argument("--model_deconv_filters", type=tuple, help="Tuple determining the 3 layer deconv"
                                                                   "filter size for the model. For example"
                                                                   "[64, 32, 16]")
    parser.add_argument("--dropout_prob", type=float, help="Determines the dropout probability for"
                                                           "Dropout layers in the model. For example 0.1")
    parser.add_argument("--heatmap_size", type=int, help="Size of the heatmaps that are fed into the model"
                                                         "by default its just 112 which means that the heatmaps"
                                                         "are of shape 112 x 112")
    parser.add_argument("--vit_model", type=str,
                        help="Which of the different vit models from googles pretrained models to use."
                             "Available models are: google/vit-base-patch16-224-in21k, ")

    # Dataset general
    parser.add_argument("--dataset", type=str, help="Which dataset to use for running the program: shrec")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--random_sub_micrographs", type=bool,
                        help="Randomly samples sub_heatmaps and sub micrographs instead of doing it with a "
                             "sliding window")
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
    parser.add_argument("--shrec_min_z", type=int, default=170, help="Minimum z to start taking z slices "
                                                                     "from for the shrec dataset")
    parser.add_argument("--shrec_max_z", type=int, default=340, help="Maximum z to start taking z slices "
                                                                     "from for the shrec dataset")
    # TODO: Add support for multiple particles maybe?
    parser.add_argument("--shrec_specific_particle", type=str, default=None,
                        help="If specified, the dataset will only create 3d gaussians for the specified particle.")
    parser.add_argument("--use_shrec_reconstruction", type=bool, default=False,
                        help="If set to true the program will use the reconstruction.mrc file from shrec instead of"
                             "simulating a reconstruction itself.")

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
    args.validation_loss_log_path = os.path.join(args.result_dir, "validation_loss_log.txt")

    if args.mode == "eval":
        print("Running in evaluation mode")
    elif args.mode == "train":
        print("Running in training mode")

    if args.existing_evaluation_folder != "" and args.existing_result_folder == "":
        raise Exception("You specified an existing evaluation folder but not an experiment folder, please specify the "
                        "experiment folder in which the existing evaluation folder exists.")
    if args.vit_model not in [
        'google/vit-base-patch16-224-in21k',
        'google/vit-large-patch16-224-in21k'
    ]:
        raise Exception(f"The specified ViT model {args.vit_model} is not supported.")
    if args.vit_model == 'google/vit-base-patch16-224-in21k' and args.latent_dim != 768:
        raise Exception(f"ViT Model {args.vit_model} has a latent dimension of 768, you specified {args.latent_dim}")
    if args.vit_model == 'google/vit-large-patch16-224-in21k' and args.latent_dim != 1024:
        raise Exception(f"ViT Model {args.vit_model} has a latent dimension of 1024, you specified {args.latent_dim}")
    if args.num_vit_finetune_layers > 8:
        raise Exception(f"Are you sure you want to finetune {args.num_vit_finetune_layers} layers of the ViT? "
                        f"The base 16x16 ViT only has 11 transformer layers."
                        f"If this was not accidental, you can disable this exception message")
    if args.finetune_vit and (args.num_vit_finetune_layers is None or args.num_vit_finetune_layers <= 0):
        raise Exception(f"You want to finetune the ViT but you haven't specified the number of layers that you"
                        f"want to finetune, or the number of layers is negative which makes no sense.")
    if args.use_shrec_reconstruction and not args.use_fbp:
        raise Exception(f"You want to use shrec reconstruction but didnt set use_fbp to true, this does nothing.")

    return args


def create_folders_and_initiate_files(args):
    """
    Creates necessary folders and initializes files for logging and experiment results.

    :param args: Requires the argument parser specified in main.py get_args()
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
            f.write("epoch,average_loss\n")
        with open(args.validation_loss_log_path, 'w') as f:
            f.write("epoch,average_validation_loss\n")


def main():
    args = get_args()
    create_folders_and_initiate_files(args)

    # ViT model
    vit_image_processor = ViTImageProcessor.from_pretrained(args.vit_model)
    if not args.train_vit_from_scratch:
        print(f"Loading ViT model: {args.vit_model}")
        vit_model = ViTModel.from_pretrained(args.vit_model)
    else:
        config = ViTConfig()
        vit_model = ViTModel(config)
        print(f"Training ViT model from scratch with parameters {vit_model.config}")

    vit_model.to(args.device)
    if args.mode == "eval" and (args.finetune_vit or args.train_vit_from_scratch):
        print("Loading finetuned or trained from scratch ViT model")
        vit_model.load_state_dict(torch.load(os.path.join(args.result_dir, 'vit_checkpoint_final.pth'),
                                             map_location=args.device))

    # Freeze all parameters (finetuning vit)
    for param in vit_model.parameters():
        param.requires_grad = False

    # Unfreeze last x transformer layers (finetuning vit)
    if args.finetune_vit:
        print(f"Unfreezing last {args.num_vit_finetune_layers} layers of the ViT for finetuning")
        for block in vit_model.encoder.layer[-args.num_vit_finetune_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    # Dataset
    dataset = ShrecDataset(sampling_points=args.shrec_sampling_points, z_slice_size=args.shrec_z_slice_size,
                           model_number=args.shrec_model_number, min_z=args.shrec_min_z, max_z=args.shrec_max_z,
                           particle_height=args.particle_height, particle_width=args.particle_width,
                           particle_depth=args.particle_depth, noise=args.noise, add_noise=args.add_noise,
                           device=args.device, use_fbp=args.use_fbp, fbp_min_angle=args.fbp_min_angle,
                           fbp_max_angle=args.fbp_max_angle, fbp_num_projections=args.fbp_num_projections,
                           shrec_specific_particle=args.shrec_specific_particle, heatmap_size=args.heatmap_size,
                           random_sub_micrographs=args.random_sub_micrographs,
                           use_shrec_reconstruction=args.use_shrec_reconstruction)

    validation_dataset = ShrecDataset(
        sampling_points=args.shrec_sampling_points, z_slice_size=args.shrec_z_slice_size,
        model_number=args.shrec_validation_model_number,  # Important difference, we use the validation volumes
        min_z=args.shrec_min_z, max_z=args.shrec_max_z, particle_height=args.particle_height,
        particle_width=args.particle_width, particle_depth=args.particle_depth, noise=args.noise,
        add_noise=args.add_noise, device=args.device, use_fbp=args.use_fbp, fbp_min_angle=args.fbp_min_angle,
        fbp_max_angle=args.fbp_max_angle, fbp_num_projections=args.fbp_num_projections,
        shrec_specific_particle=args.shrec_specific_particle, heatmap_size=args.heatmap_size,
        random_sub_micrographs=False, use_shrec_reconstruction=args.use_shrec_reconstruction)

    # We only need to create the split file if were training, otherwise we read from it
    train_dataloader, test_dataloader, validation_dataloader = prepare_dataloaders(
        dataset=dataset, batch_size=args.batch_size, validation_dataset=validation_dataset)

    model = TopdownHeatmapSimpleHead(in_channels=args.latent_dim, out_channels=1,
                                     num_deconv_filters=tuple(args.model_deconv_filters),
                                     dropout_prob=args.dropout_prob)
    model.init_weights()
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of trainable parameters: {num_params}")
    print(f"Training Dataset size: {dataset.__len__()}")
    print(f"Validation Dataset size: {validation_dataset.__len__()}")

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
            vit_model.train()

        last_checkpoint_time = time.time()

        # Save untrained checkpoint for debugging purposes
        torch.save(model.state_dict(), os.path.join(args.result_dir, f'checkpoints/checkpoint_untrained.pth'))

        plotted = 0
        best_validation_loss = float('inf')
        patience_counter = 0
        for epoch in range(args.epochs):
            epoch_bar = tqdm(range(len(train_dataloader)), desc=f'Epoch [{epoch + 1}/{args.epochs}]', unit='batch')
            # This is different from batch_index as this only counts how many batches have been done since the last
            # avg_loss calculation
            batch_counter = 0
            running_loss = 0.0
            validation_batch_counter = 0
            running_validation_loss = 0.0

            # Training here
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
                # TODO: technically you could adjust the 14, 14 to be calculated but its unnecessary as long as you
                #  don't change the vit input size
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

                if time.time() - last_checkpoint_time > args.checkpoint_interval:
                    # Save checkpoint
                    torch.save(model.state_dict(), os.path.join(
                        args.result_dir, f'checkpoints/checkpoint_epoch_{epoch}_batch_{batch_index}.pth'))

                    if args.finetune_vit:
                        torch.save(vit_model.state_dict(), os.path.join(
                            args.result_dir, f"checkpoints/vit_checkpoint_epoch{epoch}_batch_{batch_index}.pth"))

                    last_checkpoint_time = time.time()

            avg_loss = running_loss / batch_counter
            # Save running loss to log file
            with open(args.loss_log_path, 'a') as f:
                f.write(f"{epoch}, {avg_loss}\n")

            epoch_bar.close()
            if args.random_sub_micrographs:
                dataset.update_sub_micrographs()

            # Validation here
            with torch.no_grad():
                for batch_index, (micrographs, target_heatmaps, target_coordinates_list, debug_tuple) \
                        in enumerate(validation_dataloader):
                    validation_batch_counter += 1

                    encoded_image = get_encoded_image(micrographs, vit_model, vit_image_processor, device=args.device)
                    # 1: here because we don't need the class token
                    latent_micrographs = encoded_image['last_hidden_state'].to(args.device)[:, 1:, :]
                    # Right shape for model, we permute the hidden dimension to the second place
                    # TODO: technically you could adjust the 14, 14 to be calculated but its unnecessary as long as you
                    #  don't change the vit input size
                    latent_micrographs = latent_micrographs.permute(0, 2, 1)
                    latent_micrographs = latent_micrographs.reshape(latent_micrographs.size(0),
                                                                    latent_micrographs.size(1), 14, 14)
                    outputs = model(latent_micrographs)

                    validation_losses = mse_loss(outputs["heatmaps"], target_heatmaps)
                    running_validation_loss += validation_losses.item()
            avg_validation_loss = running_validation_loss / validation_batch_counter
            # Save running validation loss to log file
            with open(args.validation_loss_log_path, 'a') as f:
                f.write(f"{epoch},{avg_validation_loss}\n")

            # Plot the loss log after every epoch
            plot_loss_log(args.loss_log_path, args.validation_loss_log_path, args.result_dir)

            print(f'\nEpoch {epoch}: Train Loss {avg_loss}, Eval Loss {avg_validation_loss}, best validation loss {best_validation_loss}, patience counter {patience_counter}')
            if avg_validation_loss < best_validation_loss:
                print(f"Found new best loss, patience counter reset")
                best_validation_loss = avg_validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > args.patience:
                print(f"Early stopping triggered, validation loss has not improved in the last {args.patience} epochs.")
                break

        # Save final checkpoint
        torch.save(model.state_dict(), os.path.join(args.result_dir, f"checkpoint_final.pth"))
        if args.finetune_vit:
            torch.save(vit_model.state_dict(), os.path.join(args.result_dir, f"vit_checkpoint_final.pth"))

        # Plot the loss log after training
        plot_loss_log(args.loss_log_path, args.validation_loss_log_path, args.result_dir)

    if args.mode == "eval":
        model.eval()
        vit_model.eval()
        if not os.path.exists(os.path.join(args.result_dir, "losses_plot.png")):
            plot_loss_log(args.loss_log_path, args.validation_loss_log_path, args.result_dir)

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
