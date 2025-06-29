import torch
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_folder = "ablation_studies_results"
save_folder = "plots"


def plot_roc_curve():
    with open(os.path.join(results_folder, "studying_thresholds_for_recall_and_precision.json"), "r") as f:
        data = json.load(f)

    avg_precision = np.array(data["avg_precision"])
    avg_recall = np.array(data["avg_recall"])

    tpr = avg_recall

    fpr = 1 - avg_precision

    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    auc = np.trapz(tpr_sorted, fpr_sorted)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_sorted, tpr_sorted, marker='o', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

    plt.xlabel('False Positive Rate (1 - Precision)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "roc_curve.png"), dpi=300)
    plt.close()


def plot_precision_recall_threshold():
    with open(os.path.join(results_folder, "studying_thresholds_for_recall_and_precision.json"), "r") as f:
        data = json.load(f)

    prediction_thresholds = data["prediction_thresholds"]
    avg_precision = data["avg_precision"]
    avg_recall = data["avg_recall"]

    plt.figure(figsize=(10, 6))
    plt.plot(prediction_thresholds, avg_precision, marker='o', label='Precision', linewidth=2)
    plt.plot(prediction_thresholds, avg_recall, marker='s', label='Recall', linewidth=2)

    plt.xlabel('Prediction Threshold')
    plt.ylabel('Score')
    plt.title('Effect of Prediction Threshold on Precision and Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "precision_recall_threshold.png"), dpi=300)
    plt.close()


def plot_f1_scores_parameters():
    models = ["base_model_0dB", "base_model_10dB", "base_model_-10dB", "base_model_-5dB", "shrec_reconstruction_volume"]

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
        plt.title(f'Parameter Comparison, {model}')
        plt.savefig(os.path.join(save_folder, f"parameters_{model}.png"))
        plt.close()


def plot_vit_models():
    models = ["vit_models"]

    for model in models:
        df = pd.read_csv(os.path.join(results_folder, f"{model}.txt"))

        df['vit_model_name'] = df['vit_model_name'].astype(str)

        plt.plot(df['vit_model_name'], df['avg_f1_score'], marker='o', label='F1 Score')
        plt.plot(df['vit_model_name'], df['avg_precision'], marker='o', label='Precision')
        plt.plot(df['vit_model_name'], df['avg_recall'], marker='o', label='Recall')

        plt.xlabel("Vit Model Name")
        plt.ylabel("Score")
        plt.title("ViT model ablation study")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{model}.png"))
        plt.close()


def plot_finetuning():
    models = ["finetuning_vit", "new_finetuning_vit"]

    for model in models:
        df = pd.read_csv(os.path.join(results_folder, f"{model}.txt"))

        plt.plot(df['finetuning_layers'], df['avg_f1_score'], marker='o', label='F1 Score')
        plt.plot(df['finetuning_layers'], df['avg_precision'], marker='o', label='Precision')
        plt.plot(df['finetuning_layers'], df['avg_recall'], marker='o', label='Recall')

        plt.xlabel('Finetuned Layers')
        plt.ylabel('Score')
        plt.title('Number of finetuned layers ablation study')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, f"{model}.png"))
        plt.close()


def plot_model_size():
    models = ["model_size", "new_model_size"]

    for model in models:
        df = pd.read_csv(os.path.join(results_folder, f"{model}.txt"))

        df['filters'] = df['model_deconvolution_filters'].astype(str)

        plt.plot(df['filters'], df['avg_f1_score'], marker='o', label='F1 Score')
        plt.plot(df['filters'], df['avg_precision'], marker='o', label='Precision')
        plt.plot(df['filters'], df['avg_recall'], marker='o', label='Recall')

        plt.xlabel("Model Deconvolution Filters")
        plt.ylabel("Score")
        plt.title("Deconvolution filters ablation studies")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, f"{model}.png"))
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
    plot_vit_models()
    plot_precision_recall_threshold()
    plot_roc_curve()


if __name__ == "__main__":
    main()
