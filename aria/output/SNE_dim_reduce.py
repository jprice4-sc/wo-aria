import base64
import io
import os

import cv2
import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def encode_image(image_path):
    """Encode an image as a base64 string after clipping values to 0-2000."""

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image not found or cannot be read.")

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_clipped = np.clip(img, 0, 2000)

    # Normalize
    img_normalized = ((img_clipped / 2000) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_normalized)

    # encode the image as a base64 string
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{encoded_img}"


def plot_tsne_with_dbscan(
    input_csv_path,
    output_csv_path,
    num_dimensions=2,
    perplexity=30,
    n_iter=1000,
    eps=0.5,
    min_samples=5,
    save_csv=False,
):
    """Plot t-SNE reduced dimensions with DBSCAN clustering and display images on click."""

    df = pd.read_csv(input_csv_path)
    # Extract latent dims
    latent_columns = [col for col in df.columns if col.startswith("Latent Dim")]
    latent_data = df[latent_columns].values

    # tSNE
    tsne = TSNE(n_components=num_dimensions, perplexity=perplexity, n_iter=n_iter, random_state=0)
    reduced_data = tsne.fit_transform(latent_data)

    for i in range(num_dimensions):
        df[f"t-SNE Dim {i}"] = reduced_data[:, i]

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(reduced_data)

    if save_csv:
        output_csv_path = output_csv_path
        df.to_csv(output_csv_path, index=False)

    # Plot
    if num_dimensions == 2:
        fig = px.scatter(
            df,
            x="t-SNE Dim 0",
            y="t-SNE Dim 1",
            color="cluster",
            hover_name="Image Path",
            title="2D t-SNE with DBSCAN Clustering",
        )
    elif num_dimensions == 3:
        fig = px.scatter_3d(
            df,
            x="t-SNE Dim 0",
            y="t-SNE Dim 1",
            z="t-SNE Dim 2",
            color="cluster",
            hover_name="Image Path",
            title="3D t-SNE with DBSCAN Clustering",
        )
    else:
        raise ValueError("Currently, only 2D and 3D plots are supported.")

    fig.update_traces(marker=dict(size=3))
    app = dash.Dash(__name__)
    app.layout = html.Div([dcc.Graph(id="tsne-dbscan-plot", figure=fig), html.Div(id="image-container")])

    @app.callback(Output("image-container", "children"), [Input("tsne-dbscan-plot", "clickData")])
    def display_image(clickData):
        if clickData is not None:
            image_path = clickData["points"][0]["hovertext"]
            if os.path.exists(image_path):
                encoded_image = encode_image(image_path)
                return html.Img(src=encoded_image, style={"height": "300px"})
        return "Click on a point to display the image."

    app.run_server(debug=True)


if __name__ == "__main__":
    input_csv_path = r"D:\ADA\latent_coords\latent_coordinates_conv_flat_k_post_sept.csv"
    output_csv_path = "tsne_dbscan.csv"
    plot_tsne_with_dbscan(
        input_csv_path,
        output_csv_path,
        num_dimensions=2,
        perplexity=20,
        n_iter=1000,
        eps=3.7,
        min_samples=10,
        save_csv=False,
    )
