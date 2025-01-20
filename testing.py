import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class ParticleDetector(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(ParticleDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)  # Output layer for coordinates

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = ParticleDetector()
criterion = nn.MSELoss()  # Assuming you have ground truth coordinates
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for latent_repr, target_coords in dataloader:
            optimizer.zero_grad()
            outputs = model(latent_repr)
            loss = criterion(outputs, target_coords)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Example usage with dummy data
# Assuming dataloader provides (latent_repr, target_coords) pairs
# latent_repr: Tensor of shape (batch_size, 768)
# target_coords: Tensor of shape (batch_size, 2)
# dataloader = ...

# train(model, dataloader, criterion, optimizer)