import torch
import torch.nn as nn
import numpy as np


def load_vit_model():
    # TODO: actually use this
    pass


class ParticlePicker(nn.Module):
    def __init__(self, latent_dim, num_particles, image_width, image_height):
        super(ParticlePicker, self).__init__()

        self.width = image_width
        self.height = image_height
        self.latent_dim = latent_dim
        self.num_particles = num_particles

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Output N num_particles x (x, y, width, height, prob_no_particle, prob_yes_particle) coordinates
            nn.Linear(256, num_particles * 6)
        )

    def forward(self, x):
        """
        :param x: Input
        :return:
        """

        out = self.fc(x).view(-1, self.num_particles, 6)
        out[:, :, :4] = torch.sigmoid(out[:, :, :4])  # Box coordinates
        return out
