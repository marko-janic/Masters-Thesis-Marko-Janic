import torch
import unittest
import os
import argparse
import random

import matplotlib.pyplot as plt

import torch.optim as optim
from scipy.optimize import linear_sum_assignment  # Add this import

# Local imports
from loss import build
from main import get_args
from train import prepare_outputs_for_loss
from util.utils import create_folder_if_missing

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
EPOCHS = 1000
LEARNING_RATE = 0.1
NUM_TARGETS = 3
NUM_PREDICTIONS = 5
TEST_RESULTS_FOLDER = 'test_loss'

MIN_X_TARGET = 60
MIN_Y_TARGET = 60
PARTICLE_WIDTH = 10
PARTICLE_HEIGHT = 10
X_INIT = 0
Y_INIT = 0
PARTICLE_WIDTH_INIT = 10
PARTICLE_HEIGHT_INIT = 10


class LossTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.args = get_args()
        self.criterion, self.postprocessors = build(self.args)

        coordinates = torch.Tensor([[random.randint(60, IMAGE_WIDTH), random.randint(60, IMAGE_HEIGHT)] for _ in range(NUM_TARGETS)])
        particle_sizes = torch.tensor([PARTICLE_WIDTH, PARTICLE_HEIGHT],
                                      dtype=torch.float32).repeat(len(coordinates), 1)
        bboxes = (torch.cat((coordinates, particle_sizes), dim=1) / 
                  torch.Tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]))
        classes = torch.zeros(len(bboxes))  # Zeros since we have one class (particle)
        self.target = [{
            "boxes": bboxes.to(self.args.device),
            "labels": classes.to(self.args.device),
            "orig_size": torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT]).to(self.args.device),
            "image_id": 0
        }]

    def test_loss_gradient(self):
        model = torch.rand((1, NUM_PREDICTIONS, 6))
        model[:, :, 0] = X_INIT
        model[:, :, 1] = Y_INIT
        model[:, :, 2] = PARTICLE_WIDTH_INIT / IMAGE_WIDTH
        model[:, :, 3] = PARTICLE_HEIGHT_INIT / IMAGE_HEIGHT
        model.requires_grad_()
        optimizer = optim.Adam([model], lr=LEARNING_RATE)

        losses_list = []
        mse_list = []

        for i in range(EPOCHS):
            outputs = prepare_outputs_for_loss(model)

            # Set negative width or height values to 0
            clamped_boxes = torch.clamp(outputs['pred_boxes'][:, :, 2:4], min=0)
            outputs['pred_boxes'] = torch.cat((outputs['pred_boxes'][:, :, :2], clamped_boxes, outputs['pred_boxes'][:, :, 4:]), dim=2)

            loss_dict = self.criterion(outputs, self.target)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            losses_list.append(losses.item())

            # Calculate distances between target centers and predictions
            target_centers = self.target[0]['boxes'][:, :2] * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT])
            prediction_centers = model[0, :, :2] * torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT])
            distances = torch.cdist(target_centers.unsqueeze(0), prediction_centers.unsqueeze(0)).squeeze(0)
            
            # Use Hungarian algorithm for optimal matching
            row_ind, col_ind = linear_sum_assignment(distances.cpu().detach().numpy())
            valid_predictions = model[:, col_ind, :4]
            mse = torch.mean((self.target[0]['boxes'] - valid_predictions) ** 2).item()
            mse_list.append(mse)

            if i == EPOCHS - 1:
                print(f"Epoch {i + 1}, Loss: {losses.item()}")
                print(f"Targets: \n{self.target[0]['boxes']}")
                print(f"Matched Predictions: \n{valid_predictions}")
                print(f"Model: {model}")

        plt.figure(figsize=(7, 10))  # Adjust the figure size
        plt.subplot(3, 1, 1)
        plt.semilogy(losses_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'(Particle Size: {PARTICLE_WIDTH}x{PARTICLE_HEIGHT}) Final Loss: {losses_list[-1]:.4f}')
        plt.subplot(3, 1, 2)
        plt.semilogy(mse_list)
        plt.xlabel('Epochs')
        plt.ylabel('MSE predictions vs target coords')

        # Add a table with the parameters
        plt.subplot(3, 1, 3)
        plt.axis('off')
        table_data = [
            ["MIN_X_TARGET", MIN_X_TARGET],
            ["MIN_Y_TARGET", MIN_Y_TARGET],
            ["PARTICLE_WIDTH", PARTICLE_WIDTH],
            ["PARTICLE_HEIGHT", PARTICLE_HEIGHT],
            ["X_INIT", X_INIT],
            ["Y_INIT", Y_INIT],
            ["PARTICLE_WIDTH_INIT", PARTICLE_WIDTH_INIT],
            ["PARTICLE_HEIGHT_INIT", PARTICLE_HEIGHT_INIT]
        ]
        table = plt.table(cellText=table_data, colLabels=["Parameter", "Value"], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.savefig(os.path.join(TEST_RESULTS_FOLDER, 'loss_plot.png'))


if __name__ == '__main__':
    unittest.main()
