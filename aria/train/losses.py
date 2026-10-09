import torch
import torch.nn.functional as F


def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x, y, window_size=11, sigma=1.5, data_range=1.0):
    """Mean structural similarity between two (N, C, H, W) batches in [0, data_range]."""
    c = x.shape[1]
    window = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    sigma_x = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x.pow(2)
    sigma_y = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y.pow(2)
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def reconstruction_loss(x_hat, x, ssim_weight=0.85):
    """L1 + SSIM. SSIM is normalised by local contrast, so faint texture (the honeycomb pattern) is penalised
    as heavily as large-scale shading, which plain MSE would happily blur away.
    """
    return (1 - ssim_weight) * F.l1_loss(x_hat, x) + ssim_weight * (1 - ssim(x_hat, x))
