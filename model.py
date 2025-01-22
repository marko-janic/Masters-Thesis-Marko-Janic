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
            nn.Linear(256, num_particles * 3)  # Output N x (x, y) coordinates
        )

    def forward(self, x):
        return self.fc(x).view(-1, self.num_particles, 3)  # Reshape to (batch_size, N, 3)
