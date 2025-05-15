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
    crop_fov: Optional[List[float]] = None,
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
    crop_fov: List[List[float], List[float]]
        Crop the masks and points
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
    # TODO implement cropping, filter points in / outside range
    # crop_fov =
    viewer.add_points(
        points,
        features={"percentage": np.round(percentage_values, 1)},
        text={"string": "{percentage}%", "color": "white", **textkwargs},
        size=0.01,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return


def pandas_df_to_napari_tracks(
    df: pd.DataFrame,
    viewer: napari.Viewer,
    unique_track_id_name: str,
    frame_id_name: str,
    position_x_name: str,
    position_y_name: str,
    feature_name: Optional[str] = None,
    colormaps_dict: Optional[dict] = None,
) -> None:
    """Add tracks to Napari track layer.
    Splitting and merging are not yet implemented.
    """
    # filter for valid tracks only
    track_df = df[df[unique_track_id_name] > 0]
    track_data = track_df[
        [unique_track_id_name, frame_id_name, position_y_name, position_x_name]
    ].to_numpy()
    if track_df[feature_name].min() < 0 or track_df[feature_name].max() > 100.0:
        raise ValueError(
            "Make sure that the features are between 0 and 1, "
            "otherwise the colormapping does not work well"
        )
    features = None
    if feature_name is not None:
        features = {feature_name: track_df[feature_name].to_numpy()}
    viewer.add_tracks(track_data, features=features, colormaps_dict=colormaps_dict)
