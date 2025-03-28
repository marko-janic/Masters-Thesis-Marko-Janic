import torch
import torch.nn as nn


class ParticlePicker(nn.Module):
    def __init__(self, latent_dim: int, num_particles: int, image_width: int, image_height: int,
                 num_patch_embeddings: int, include_class_token: bool, only_use_class_token: bool):
        super(ParticlePicker, self).__init__()

        self.width = image_width
        self.height = image_height
        self.latent_dim = latent_dim
        self.num_particles = num_particles
        self.num_patch_embeddings = num_patch_embeddings
        self.include_class_token = include_class_token
        self.only_use_class_token = only_use_class_token

        if self.only_use_class_token:
            self.initial_input_size = self.latent_dim
        else:
            if self.include_class_token:
                self.initial_input_size = self.latent_dim * (self.num_patch_embeddings + 1)
            else:
                self.initial_input_size = self.latent_dim * self.num_patch_embeddings

        self.fc = nn.Sequential(
            nn.Linear(self.initial_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Output N num_particles x (x, y, width, height, prob_no_particle, prob_yes_particle) coordinates
            nn.Linear(256, num_particles * 6)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape batch x num_patch_embeddings + 1 x latent_dimension. The + 1 refers to the cls token
        :return: particle predictions in shape N x (x, y, width, height, prob_no_particle, prob_yes_particle)
        """

        if not self.include_class_token and not self.only_use_class_token:
            x = x[:, 1:]  # We don't want the class token here (it is located at position 0)

        # Flatten the input tensor to shape (batch_size, initial_input_size)
        x = x.view(x.size(0), -1)

        out = self.fc(x).view(-1, self.num_particles, 6)
        out[:, :, :4] = torch.sigmoid(out[:, :, :4])  # Box coordinates
        return out
