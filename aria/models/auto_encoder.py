# vae_components.py
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dims, image_size):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.linear1 = nn.Linear(image_size * image_size * 1, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims, image_size):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, image_size * image_size * 1)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.view(-1, 1, self.image_size, self.image_size)


class Autoencoder(nn.Module):
    def __init__(self, latent_dims, image_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, image_size)
        self.decoder = Decoder(latent_dims, image_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape).to(x.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
