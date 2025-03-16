import torch
import unittest
import os
import argparse
import random

import matplotlib.pyplot as plt

import torch.optim as optim

# Local imports
from loss import build
from main import get_args
from train import prepare_outputs_for_loss
from util.utils import create_folder_if_missing

TEST_RESULTS_FOLDER = 'test_loss'
PARTICLE_WIDTH = 80
PARTICLE_HEIGHT = 80
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
EPOCHS = 500
LEARNING_RATE = 0.1
NUM_TARGETS = 3
NUM_PREDICTIONS = 5


class LossTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.args = get_args()
        self.criterion, self.postprocessors = build(self.args)

        coordinates = torch.Tensor([[random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)] for _ in range(NUM_TARGETS)])
        particle_sizes = torch.tensor([PARTICLE_WIDTH, PARTICLE_HEIGHT],
                                      dtype=torch.float32).repeat(len(coordinates), 1)
        bboxes = (torch.cat((coordinates, particle_sizes), dim=1) /
                  torch.Tensor([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]))
        classes = torch.zeros(len(bboxes))  # Zeros since we have one class (particle)
        self.target = [{
            "boxes": bboxes,
            "labels": classes,
            "orig_size": torch.tensor([IMAGE_WIDTH, IMAGE_HEIGHT]),
            "image_id": 0
        }]

    def test_loss_gradient(self):
        model = torch.rand((1, NUM_PREDICTIONS, 6))
        model.requires_grad_()
        optimizer = optim.Adam([model], lr=LEARNING_RATE)

        losses_list = []
        mse_list = []

        for i in range(EPOCHS):
            outputs = prepare_outputs_for_loss(model)

            loss_dict = self.criterion(outputs, self.target)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            losses_list.append(losses.item())

            topk_indices = torch.topk(model[0, :, 4], NUM_TARGETS).indices
            valid_predictions = model[:, topk_indices, :4]
            mse = torch.mean((self.target[0]['boxes'] - valid_predictions) ** 2).item()
            mse_list.append(mse)

            if i == EPOCHS - 1:
                print(f"Epoch {i + 1}, Loss: {losses.item()}")
                print(f"Targets: {self.target}")
                print(f"Model: {model}")

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(losses_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.subplot(2, 1, 2)
        plt.plot(mse_list)
        plt.xlabel('Epochs')
        plt.ylabel('MSE predictions vs target coords')
        plt.savefig(os.path.join(TEST_RESULTS_FOLDER, 'loss_plot.png'))


if __name__ == '__main__':
    unittest.main()
