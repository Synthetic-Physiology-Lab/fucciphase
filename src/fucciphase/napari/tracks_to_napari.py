from typing import List, Optional

import numpy as np
import pandas as pd

HAS_NAPARI = True
try:
    import napari
except ImportError:
    HAS_NAPARI = False
    pass


def add_trackmate_data_to_viewer(
    df: pd.DataFrame,
    viewer: napari.Viewer,
    scale: tuple,
    image_data: List[np.ndarray],
    colormaps: List[str],
    labels: Optional[np.ndarray],
    dim: int = 2,
) -> None:
    """Overlay tracking result and video.

    df: pd.DataFrame
        TrackMate result processed by fucciphase
    viewer: napari.Viewer
        Viewer instance
    scale: tuple
        Pixel sizes as tuple of size dim
    image_data: np.ndarray
        List of image arrays
    colormaps: List[str]
        List of colormaps for each image channel
    labels: Optional[np.ndarray]
        Segmentation masks
    """
    if not HAS_NAPARI:
        raise ImportError("Please install napari")
    if dim != 2:
        raise NotImplementedError("Workflow currently only implemented for 2D frames.")
    # make sure it is sorted
    napari_val_df = df.sort_values("POSITION_T")
    # extract points
    points = napari_val_df[["POSITION_T", "POSITION_Y", "POSITION_X"]].to_numpy()
    # extract percentages at points
    percentage_values = napari_val_df["CELL_CYCLE_PERC_POST"].to_numpy()
    if labels is not None:
        new_labels = np.zeros(shape=labels.shape, dtype=labels.dtype)
        # add labels to each frame
        for i in range(round(df["POSITION_T"].max()) + 1):
            subdf = df[np.isclose(df["POSITION_T"], i)]
            label_ids = subdf["MEAN_INTENSITY_CH3"]
            track_ids = subdf["TRACK_ID"]
            for idx, label_id in enumerate(label_ids):
                # add 1 because TRACK_ID 0 would be background
                new_labels[i][np.isclose(labels[i], label_id)] = track_ids.iloc[idx] + 1
        labels_layer = viewer.add_labels(labels, scale=scale)
        labels_layer.contour = 10

    for image, colormap in zip(image_data, colormaps):
        viewer.add_image(image, blending="additive", colormap=colormap, scale=scale)

    viewer.add_points(
        points,
        features={"percentage": np.round(percentage_values, 1)},
        text={"string": "{percentage}%", "color": "white"},
        size=0.01,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return
