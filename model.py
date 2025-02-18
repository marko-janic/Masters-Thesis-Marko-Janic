import torch
import torch.nn as nn
import numpy as np


class ParticlePicker(nn.Module):
    def __init__(self, latent_dim, num_particles):
        self.latent_dim = latent_dim
        self.num_particles = num_particles

        super(ParticlePicker, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Output N num_particles x (x, y, box_width, box_height, prob_no_particle, prob_yes_particle) coordinates
            nn.Linear(256, num_particles * 6)
        )

    def forward(self, x):
        """
        :param x: Input
        :return: A tensor of size batch_size x num_particles x 3 where the last column of a specific particle
        is its class
        """
        out = self.fc(x).view(-1, self.num_particles, 6)  # Reshape to (batch_size, N, 4)
        out[:, :, 4:] = torch.sigmoid(out[:, :, 4:])  # Presence probability
        return out
