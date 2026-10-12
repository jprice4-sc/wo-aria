import base64
import io
import os

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from dash import ALL, Dash, Input, Output, ctx, dcc, html
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


class LatentPCA:
    """PCA over the full flattened latents, so any vector of PC coordinates maps back to a full latent.

    Coordinates are in units of each component's standard deviation, so 0 is the mean latent and
    +/-2 is the typical spread of the data along that component.
    """

    def __init__(self, latents, n_components=20):
        self.latent_shape = tuple(latents.shape[1:])
        flat = latents.flatten(start_dim=1).numpy()
        self.pca = PCA(n_components=min(n_components, *flat.shape), svd_solver="randomized", random_state=0).fit(flat)
        self.sd = np.sqrt(self.pca.explained_variance_)
        self.explained = self.pca.explained_variance_ratio_

    @property
    def n_components(self):
        return len(self.sd)

    def project(self, latents):
        """(N, C, H, W) latents -> (N, K) coordinates in SD units."""
        return self.pca.transform(latents.flatten(start_dim=1).numpy()) / self.sd

    def to_latent(self, coords):
        """(K,) or (N, K) coordinates in SD units -> (N, C, H, W) latent tensor."""
        coords = np.atleast_2d(coords) * self.sd
        flat = self.pca.inverse_transform(coords).astype(np.float32)
        return torch.from_numpy(flat).reshape(-1, *self.latent_shape)


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
            x=gx.ravel(),
            y=gy.ravel(),
            mode="markers",
            marker=dict(size=8, opacity=0),
            hoverinfo="none",
            showlegend=False,
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


def build_pca_sliders(pca, slider_range=3.0):
    sliders = []
    for i in range(pca.n_components):
        sliders.append(html.Label(f"PC{i + 1}  ({pca.explained[i]:.1%} of variance)"))
        sliders.append(
            dcc.Slider(
                id={"type": "pc", "index": i},
                min=-slider_range,
                max=slider_range,
                step=0.1,
                value=0,
                marks={-slider_range: f"-{slider_range:g}σ", 0: "0", slider_range: f"+{slider_range:g}σ"},
                tooltip={"placement": "right"},
            )
        )
    return sliders


def build_app(autoencoder, latents, coords, paths, device, k=4, pca=None, slider_range=3.0):
    app = Dash(__name__)
    image_style = {"width": "420px", "margin": "8px"}
    pca = pca or LatentPCA(latents)
    total_explained = pca.explained.sum()

    app.layout = html.Div(
        [
            html.Div(
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
            ),
            html.Hr(),
            html.H3("Construct from latent PCA coordinates"),
            html.P(
                f"Each slider moves along one principal direction of the full latent space "
                f"({pca.n_components} components, {total_explained:.0%} of latent variance). "
                "No real images are used; the latent is built from these coordinates alone. "
                "Clicking the plot above sets the sliders to that point's projection."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("Reset to mean", id="pca-reset", style={"marginRight": "8px"}),
                            html.Button("Random sample", id="pca-random"),
                            html.Div(build_pca_sliders(pca, slider_range), style={"marginTop": "12px"}),
                        ],
                        style={"width": "45%", "paddingRight": "24px"},
                    ),
                    html.Div([html.Img(id="pca-image", style=image_style), html.Pre(id="pca-info")]),
                ],
                style={"display": "flex"},
            ),
        ]
    )

    @app.callback(
        Output("decoded", "src"),
        Output("nearest", "src"),
        Output("info", "children"),
        Input("latent-plot", "clickData"),
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

    @app.callback(
        Output({"type": "pc", "index": ALL}, "value"),
        Input("latent-plot", "clickData"),
        Input("pca-reset", "n_clicks"),
        Input("pca-random", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_sliders(click_data, _, n_random):
        if ctx.triggered_id == "pca-reset":
            values = np.zeros(pca.n_components)
        elif ctx.triggered_id == "pca-random":
            # coordinates are in SD units, so a standard normal draw matches the data's spread
            values = np.random.default_rng(n_random).standard_normal(pca.n_components)
        else:
            point = (click_data["points"][0]["x"], click_data["points"][0]["y"])
            z, _, _ = latent_at(point, coords, latents, k=k)
            values = pca.project(z)[0]
        return np.clip(values, -slider_range, slider_range).round(1).tolist()

    @app.callback(
        Output("pca-image", "src"), Output("pca-info", "children"), Input({"type": "pc", "index": ALL}, "value")
    )
    def decode_pca(values):
        values = np.asarray(values, dtype=np.float64)
        with torch.no_grad():
            decoded = autoencoder.decoder(pca.to_latent(values).to(device)).squeeze().cpu().numpy()
        # Mahalanobis distance from the mean in PC space; large values are far outside the training data
        info = f"distance from mean: {np.linalg.norm(values):.2f} (typical image ≈ {np.sqrt(pca.n_components):.1f})"
        return to_png(decoded), info

    return app


if __name__ == "__main__":
    run_dir = "/Users/jprice/Documents/aria/Autoencoder_work/flat_k_IQ_20261009_141629"
    model_path = os.path.join(run_dir, "spatial_vae_4_512.pth")
    preprocessed_file = "/Users/jprice/Documents/aria/preprocessed_image_tensors/IQtester_512.pt"
    latent_channels = 4
    variational = True
    method = "tsne"  # "pca" keeps global distances more faithful, "tsne" separates clusters better
    k = 4  # number of nearest images blended for a clicked point
    n_components = 20  # number of PCA sliders
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    autoencoder = load_model(model_path, latent_channels, variational, device)
    latents, paths = encode_dataset(autoencoder, prepare_data_loader(preprocessed_file, batch_size=8), device)
    coords = embed_2d(latents, method=method)
    pca = LatentPCA(latents, n_components=n_components)
    build_app(autoencoder, latents, coords, paths, device, k=k, pca=pca).run(debug=False)
