import torch
import torch.nn.functional as F
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(256 * 32 * 32, latent_dims)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dims, 256 * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 256, 32, 32)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(latent_dims)
        self.decoder = ConvDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
