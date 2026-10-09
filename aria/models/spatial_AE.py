import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels):
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Upsample(nn.Module):
    """Nearest-neighbour upsample followed by a conv; avoids ConvTranspose checkerboard artefacts."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class SpatialEncoder(nn.Module):
    """Fully convolutional encoder. Output is a (latent_channels, H / 2**(len(channels)-1), W / ...) feature map,
    so spatial layout (e.g. the honeycomb pattern position) is kept instead of squeezed through a flat vector.
    """

    def __init__(self, latent_channels=4, channels=(32, 64, 128, 256), variational=False):
        super().__init__()
        self.variational = variational
        self.kl = 0
        self.conv_in = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        layers = []
        for i, ch in enumerate(channels):
            prev = channels[max(i - 1, 0)]
            layers.append(ResBlock(prev, ch))
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
        self.down = nn.Sequential(*layers)

        self.mid = ResBlock(channels[-1], channels[-1])
        self.norm_out = _norm(channels[-1])
        out_ch = latent_channels * 2 if variational else latent_channels
        self.conv_out = nn.Conv2d(channels[-1], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.mid(self.down(self.conv_in(x)))
        h = self.conv_out(F.silu(self.norm_out(h)))
        if not self.variational:
            return h

        mu, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        # per-element mean so the weight is independent of image and latent size
        self.kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu


class SpatialDecoder(nn.Module):
    def __init__(self, latent_channels=4, channels=(32, 64, 128, 256)):
        super().__init__()
        rev = list(reversed(channels))
        self.conv_in = nn.Conv2d(latent_channels, rev[0], kernel_size=3, padding=1)
        self.mid = ResBlock(rev[0], rev[0])

        layers = []
        for i, ch in enumerate(rev):
            prev = rev[max(i - 1, 0)]
            layers.append(ResBlock(prev, ch))
            if i < len(rev) - 1:
                layers.append(Upsample(ch))
        self.up = nn.Sequential(*layers)

        self.norm_out = _norm(rev[-1])
        self.conv_out = nn.Conv2d(rev[-1], 1, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.up(self.mid(self.conv_in(z)))
        return torch.sigmoid(self.conv_out(F.silu(self.norm_out(h))))


class SpatialAutoencoder(nn.Module):
    """Convolutional (V)AE with a spatial latent. With the default 4 stages a 512x512 image maps to a
    4x64x64 latent (16x compression). Set ``variational=True`` to get a smooth latent space you can sample from.
    """

    def __init__(self, latent_channels=4, channels=(32, 64, 128, 256), variational=False):
        super().__init__()
        self.encoder = SpatialEncoder(latent_channels, channels, variational)
        self.decoder = SpatialDecoder(latent_channels, channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
