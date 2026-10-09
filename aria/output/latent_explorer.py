import base64
import io
import os

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from dash import Dash, Input, Output, dcc, html
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from aria.output.vae_visualise import load_model, prepare_data_loader
from aria.utils.image2tensor import preprocess_16bit_image


@torch.no_grad()
def encode_dataset(autoencoder, data_loader, device):
    """Returns the full spatial latents (N, C, H, W) on the CPU and their image paths."""
    latents, paths = [], []
    for x, _, path in data_loader:
        latents.append(autoencoder.encoder(x.to(device)).cpu())
        paths.extend(path)
    return torch.cat(latents), paths


def embed_2d(latents, method="tsne", pool_size=8, perplexity=20):
    """Projects latents to 2D using the same pooled features as vae_visualise."""
    features = F.adaptive_avg_pool2d(latents, pool_size).flatten(start_dim=1).numpy()
    if method == "pca":
        return PCA(n_components=2).fit_transform(features)
    return TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(features)


def latent_at(point, coords, latents, k=4):
    """2D projections can't be inverted, so blend the full latents of the k nearest images,
    weighted by inverse distance. Clicking on a data point returns (almost exactly) its own latent.
    """
    dist = np.linalg.norm(coords - np.asarray(point), axis=1)
    idx = np.argsort(dist)[:k]
    weights = 1.0 / (dist[idx] + 1e-6)
    weights /= weights.sum()
    w = torch.from_numpy(weights).float()[:, None, None, None]
    return (w * latents[idx]).sum(dim=0, keepdim=True), idx, weights


def to_png(image):
    """Encodes an (H, W) array in [0, 1] as a base64 PNG for display in the browser."""
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_figure(coords, paths, grid_size=80):
    fig = go.Figure()
    # invisible grid so clicks register in empty space, not only on data points
    pad = 0.05 * (coords.max(axis=0) - coords.min(axis=0))
    gx, gy = np.meshgrid(
        np.linspace(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0], grid_size),
        np.linspace(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1], grid_size),
    )
    fig.add_trace(
        go.Scatter(
            x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=8, opacity=0), hoverinfo="none", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(size=5),
            hovertext=[os.path.basename(p) for p in paths],
            hoverinfo="text",
            name="images",
        )
    )
    fig.update_layout(title="Click anywhere to decode that point", clickmode="event", height=700)
    return fig


def build_app(autoencoder, latents, coords, paths, device, k=4):
    app = Dash(__name__)
    image_style = {"width": "420px", "margin": "8px"}
    app.layout = html.Div(
        [
            html.Div(dcc.Graph(id="latent-plot", figure=build_figure(coords, paths)), style={"flex": "1"}),
            html.Div(
                [
                    html.H4("Decoded at clicked point"),
                    html.Img(id="decoded", style=image_style),
                    html.H4("Nearest real image"),
                    html.Img(id="nearest", style=image_style),
                    html.Pre(id="info"),
                ]
            ),
        ],
        style={"display": "flex"},
    )

    @app.callback(
        Output("decoded", "src"), Output("nearest", "src"), Output("info", "children"), Input("latent-plot", "clickData")
    )
    def decode_click(click_data):
        if click_data is None:
            return None, None, "Click on the plot."
        point = (click_data["points"][0]["x"], click_data["points"][0]["y"])
        z, idx, weights = latent_at(point, coords, latents, k=k)
        with torch.no_grad():
            decoded = autoencoder.decoder(z.to(device)).squeeze().cpu().numpy()
        nearest = preprocess_16bit_image(paths[idx[0]], target_size=decoded.shape[::-1]).squeeze().numpy()
        info = f"point: ({point[0]:.2f}, {point[1]:.2f})\n" + "\n".join(
            f"{w:.2f}  {os.path.basename(paths[i])}" for i, w in zip(idx, weights, strict=True)
        )
        return to_png(decoded), to_png(nearest), info

    return app


if __name__ == "__main__":
    run_dir = "/Users/jprice/Documents/aria/Autoencoder_work/flat_k_IQ_20261009_141629"
    model_path = os.path.join(run_dir, "spatial_vae_4_512.pth")
    preprocessed_file = "/Users/jprice/Documents/aria/preprocessed_image_tensors/IQtester_512.pt"
    latent_channels = 4
    variational = True
    method = "tsne"  # "pca" keeps global distances more faithful, "tsne" separates clusters better
    k = 4  # number of nearest images blended for a clicked point
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    autoencoder = load_model(model_path, latent_channels, variational, device)
    latents, paths = encode_dataset(autoencoder, prepare_data_loader(preprocessed_file, batch_size=8), device)
    coords = embed_2d(latents, method=method)
    build_app(autoencoder, latents, coords, paths, device, k=k).run(debug=False)
