import torch
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_folder = "ablation_studies_results"
save_folder = "plots"


def plot_f1_scores_parameters():
    models = ["base_model_0dB"]

    for model in models:
        with open(os.path.join(results_folder, f"{model}_grid_search_parameters.json"), "r") as f:
            data = json.load(f)

        prediction_thresholds = data["prediction_thresholds"]
        neighborhood_sizes = data["neighborhood_sizes"]
        avg_f1_scores = data["avg_f1_scores"]

        plt.scatter(prediction_thresholds, neighborhood_sizes, c=avg_f1_scores, cmap='viridis', s=100)
        plt.colorbar(label='f1 score')
        plt.xlabel('prediction thresholds')
        plt.ylabel('neighborhood sizes')
        plt.title(f'Avg f1 score for different parameters, {model}')
        plt.savefig(os.path.join(save_folder, f"parameters_{model}.png"))
        plt.close()


def plot_finetuning():
    df = pd.read_csv(os.path.join(results_folder, "finetuning_vit.txt"))

    plt.plot(df['finetuning_layers'], df['avg_f1_score'], marker='o', label='F1 Score')
    plt.plot(df['finetuning_layers'], df['avg_precision'], marker='o', label='Precision')
    plt.plot(df['finetuning_layers'], df['avg_recall'], marker='o', label='Recall')

    plt.xlabel('Finetuned Layers')
    plt.ylabel('Score')
    plt.title('Number of finetuned layers ablation study')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "finetuning_vit.png"))
    plt.close()


def plot_model_size():
    df = pd.read_csv(os.path.join(results_folder, "model_size.txt"))

    df['filters'] = df['model_deconvolution_filters'].astype(str)

    plt.plot(df['filters'], df['avg_f1_score'], marker='o', label='F1 Score')
    plt.plot(df['filters'], df['avg_precision'], marker='o', label='Precision')
    plt.plot(df['filters'], df['avg_recall'], marker='o', label='Recall')

    plt.xlabel("Model Deconvolution Filters")
    plt.ylabel("Score")
    plt.title("Deconvolution filters ablation studies")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "model_size.png"))
    plt.close()


def plot_noise_levels():
    df = pd.read_csv(os.path.join(results_folder, "noise_levels.txt"))

    plt.plot(df['noise'], df['avg_f1_score'], marker='o', label='F1 Score')
    plt.plot(df['noise'], df['avg_precision'], marker='o', label='Precision')
    plt.plot(df['noise'], df['avg_recall'], marker='o', label='Recall')

    plt.xlabel("Noise in dB")
    plt.ylabel("Score")
    plt.title("Levels of noise in projections ablation study")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "noise_levels.png"))
    plt.close()


def plot_training_volumes():
    df = pd.read_csv(os.path.join(results_folder, "training_volumes.txt"))

    plt.plot(df['num_training_volumes'], df['avg_f1_score'], marker='o', label='F1 Score')
    plt.plot(df['num_training_volumes'], df['avg_precision'], marker='o', label='Precision')
    plt.plot(df['num_training_volumes'], df['avg_recall'], marker='o', label='Recall')

    plt.xlabel("Number of Training Volumes")
    plt.ylabel("Score")
    plt.title("Performance Metrics vs Number of Training Volumes")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "training_volumes.png"))
    plt.close()


def main():
    plot_finetuning()
    plot_model_size()
    plot_noise_levels()
    plot_training_volumes()
    plot_f1_scores_parameters()


if __name__ == "__main__":
    main()
