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
    cycle_percentage_id: Optional[str] = "CELL_CYCLE_PERC_POST",
    dim: int = 2,
    textkwargs: Optional[dict] = None,
    label_id_name: Optional[str] = "MAX_INTENSITY_CH3",
    track_id_name: Optional[str] = "TRACK_ID",
    time_id_name: Optional[str] = "POSITION_T",
    pos_x_id_name: Optional[str] = "POSITION_X",
    pos_y_id_name: Optional[str] = "POSITION_Y",
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
    textkwargs: dict
        Dictionary to pass options to text in napari
    """
    if textkwargs is None:
        textkwargs = {}
    if not HAS_NAPARI:
        raise ImportError("Please install napari")
    if dim != 2:
        raise NotImplementedError("Workflow currently only implemented for 2D frames.")
    # make sure it is sorted
    napari_val_df = df.sort_values(time_id_name)
    # extract points
    points = napari_val_df[[time_id_name, pos_y_id_name, pos_x_id_name]].to_numpy()
    # extract percentages at points
    # TODO insert checks
    percentage_values = napari_val_df[cycle_percentage_id].to_numpy()
    if labels is not None:
        new_labels = np.zeros(shape=labels.shape, dtype=labels.dtype)
        # add labels to each frame
        for i in range(round(df[time_id_name].max()) + 1):
            subdf = df[np.isclose(df[time_id_name], i)]
            label_ids = subdf[label_id_name]
            track_ids = subdf[track_id_name]
            for idx, label_id in enumerate(label_ids):
                # add 1 because TRACK_ID 0 would be background
                new_labels[i][np.isclose(labels[i], label_id)] = track_ids.iloc[idx] + 1
        labels_layer = viewer.add_labels(new_labels, scale=scale)
        labels_layer.contour = 10

    for image, colormap in zip(image_data, colormaps):
        viewer.add_image(image, blending="additive", colormap=colormap, scale=scale)

    viewer.add_points(
        points,
        features={"percentage": np.round(percentage_values, 1)},
        text={"string": "{percentage}%", "color": "white", **textkwargs},
        size=0.01,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return
