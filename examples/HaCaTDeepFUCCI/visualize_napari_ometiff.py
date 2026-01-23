"""
Visualize FUCCIphase results in Napari using downscaled OME-TIFF video.

This script is a variant of visualize_napari.py optimized for .ome.tif files,
which are faster to load than the original .nd2 files.

Usage:
    python visualize_napari_ometiff.py --tracks processed_tracks.csv --image downscaled_hacat_100x.ome.tif
"""

import argparse
import os

import bioio
import bioio_ome_tiff
import napari
import numpy as np
import pandas as pd
import vispy.color.colormap

from fucciphase.napari import add_trackmate_data_to_viewer, pandas_df_to_napari_tracks


def load_ome_tiff_data(
    image_file: str,
    cyan_channel_idx: int = 1,
    magenta_channel_idx: int = 0,
    label_channel_idx: int | None = None,
) -> dict:
    """Load image data from OME-TIFF file.

    Parameters
    ----------
    image_file : str
        Path to OME-TIFF file
    cyan_channel_idx : int
        Channel index for cyan (default 1 for downscaled file)
    magenta_channel_idx : int
        Channel index for magenta (default 0 for downscaled file)
    label_channel_idx : int or None
        Channel index for labels, None to skip

    Returns
    -------
    dict
        Dictionary with image data and metadata
    """
    image = bioio.BioImage(image_file, reader=bioio_ome_tiff.Reader)

    # Get physical pixel sizes (may be None if not set in metadata)
    if image.physical_pixel_sizes.Y and image.physical_pixel_sizes.X:
        scale = (image.physical_pixel_sizes.Y, image.physical_pixel_sizes.X)
    else:
        # Default scale if not available in metadata
        scale = (1.0, 1.0)
        print(f"Warning: Physical pixel sizes not found in metadata, using {scale}")

    cyan = image.get_image_data("TYX", C=cyan_channel_idx)
    magenta = image.get_image_data("TYX", C=magenta_channel_idx)

    result = {
        "scale": scale,
        "cyan": cyan,
        "magenta": magenta,
        "cyan_contrast": (np.percentile(cyan, 1), np.percentile(cyan, 99.9)),
        "magenta_contrast": (np.percentile(magenta, 1), np.percentile(magenta, 99.9)),
        "n_frames": image.dims.T,
    }

    if label_channel_idx is not None:
        labels = image.get_image_data("TYX", C=label_channel_idx)
        result["labels"] = labels

    return result


def setup_viewer(viewer: napari.Viewer, dt_minutes: float = 15.0) -> None:
    """Configure viewer settings.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    dt_minutes : float
        Time step in minutes for overlay
    """
    viewer.window.resize(1200, 900)
    viewer.reset_view()

    # Scale bar
    viewer.scale_bar.unit = "um"
    viewer.scale_bar.font_size = 0
    viewer.scale_bar.ticks = False
    viewer.scale_bar.visible = True

    # Text overlay with time
    viewer.text_overlay.color = "white"
    viewer.text_overlay.blending = "translucent_no_depth"
    viewer.text_overlay.position = "top_left"
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.visible = True

    old_time = {"value": -1}

    def update_slider(event):
        time = viewer.dims.current_step[0]
        if time != old_time["value"]:
            old_time["value"] = time
            viewer.text_overlay.text = f"{round(dt_minutes * time)} min "

    viewer.dims.events.current_step.connect(update_slider)
    viewer.dims.current_step = (0, 0, 0)


def export_screenshots(
    viewer: napari.Viewer,
    frames: list[int],
    output_dir: str = ".",
    prefix: str = "frame",
) -> None:
    """Export screenshots at specific frames."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for frame in frames:
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 64
        viewer.scale_bar.visible = True
        viewer.scale_bar.font_size = 0

        viewer.dims.current_step = (frame, 0, 0)

        output_path = os.path.join(output_dir, f"{prefix}_{frame}.png")
        viewer.export_figure(output_path)
        print(f"Saved screenshot: {output_path}")


def main(
    tracks_file: str,
    image_file: str,
    screenshot_frames: list[int] | None = None,
    screenshot_dir: str = ".",
    screenshot_prefix: str = "frame",
    cyan_channel_idx: int = 1,
    magenta_channel_idx: int = 0,
    label_channel_idx: int | None = None,
    dt_minutes: float = 15.0,
    label_id_column: str = "MEDIAN_INTENSITY_CH3",
    scale_factor: float = 1.0,
) -> napari.Viewer:
    """Visualize tracks in Napari using OME-TIFF image.

    Parameters
    ----------
    tracks_file : str
        Path to processed tracks CSV
    image_file : str
        Path to OME-TIFF image file
    screenshot_frames : list of int or None
        Frame numbers to export as screenshots
    screenshot_dir : str
        Output directory for screenshots
    screenshot_prefix : str
        Prefix for screenshot filenames
    cyan_channel_idx : int
        Channel index for cyan
    magenta_channel_idx : int
        Channel index for magenta
    label_channel_idx : int or None
        Channel index for label, None to skip
    dt_minutes : float
        Time step in minutes
    label_id_column : str
        Column name for label ID
    scale_factor : float
        Factor by which the image was downscaled (used to adjust coordinates)

    Returns
    -------
    napari.Viewer
        The Napari viewer instance
    """
    print(f"Loading tracks from {tracks_file}...")
    track_df = pd.read_csv(tracks_file)

    # Ensure normalized percentage column exists
    if "CELL_CYCLE_PERC_NORM" not in track_df.columns:
        track_df["CELL_CYCLE_PERC_NORM"] = track_df["CELL_CYCLE_PERC_DTW"] / 100.0

    # Load image data
    print(f"Loading OME-TIFF from {image_file}...")
    image_data = load_ome_tiff_data(
        image_file,
        cyan_channel_idx=cyan_channel_idx,
        magenta_channel_idx=magenta_channel_idx,
        label_channel_idx=label_channel_idx,
    )

    # label data or None
    labels = image_data.get("labels", None)

    # Create viewer
    print("Creating Napari viewer...")
    viewer = napari.Viewer()
    setup_viewer(viewer, dt_minutes=dt_minutes)

    scale = image_data["scale"]
    if scale_factor != 1.0:
        print("Scaling up image.")
        scale = (scale[0] * scale_factor, scale[1] * scale_factor)

    # Add trackmate data
    add_trackmate_data_to_viewer(
        track_df,
        viewer,
        scale=scale,
        image_data=[image_data["cyan"], image_data["magenta"]],
        colormaps=["cyan", "magenta"],
        labels=labels,
        cycle_percentage_id="CELL_CYCLE_PERC_DTW",
        label_id_name=label_id_column,
        textkwargs={"size": 16},
    )

    # Set contrast limits
    if "cyan_contrast" in image_data:
        viewer.layers[1].contrast_limits = image_data["cyan_contrast"]
    if "magenta_contrast" in image_data:
        viewer.layers[2].contrast_limits = image_data["magenta_contrast"]

    # Add tracks colored by cell cycle percentage
    colormap = vispy.color.colormap.MatplotlibColormap("cool")
    pandas_df_to_napari_tracks(
        track_df,
        viewer,
        unique_track_id_name="TRACK_ID",
        frame_id_name="FRAME",
        position_x_name="POSITION_X",
        position_y_name="POSITION_Y",
        feature_name="CELL_CYCLE_PERC_NORM",
        colormaps_dict={"CELL_CYCLE_PERC_NORM": colormap},
    )

    # Style the tracks layer
    viewer.layers[-1].tail_width = 6
    viewer.layers[-1].blending = "translucent"

    if len(viewer.layers) >= 3:
        viewer.layers[-2].tail_width = 6

    # Export screenshots if requested
    if screenshot_frames:
        export_screenshots(
            viewer,
            screenshot_frames,
            output_dir=screenshot_dir,
            prefix=screenshot_prefix,
        )

    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize FUCCIphase results in Napari using OME-TIFF"
    )
    parser.add_argument(
        "--tracks",
        "-t",
        default="processed_tracks_both.csv",
        help="Processed tracks CSV file",
    )
    parser.add_argument(
        "--image",
        "-i",
        default="downscaled_hacat_100x.ome.tif",
        help="OME-TIFF image file",
    )
    parser.add_argument(
        "--cyan-channel",
        type=int,
        default=1,
        help="Cyan channel index (default: 1)",
    )
    parser.add_argument(
        "--magenta-channel",
        type=int,
        default=0,
        help="Magenta channel index (default: 0)",
    )
    parser.add_argument(
        "--labels-channel",
        type=int,
        default=2,
        help="Label channel index (default: 2)",
    )
    parser.add_argument(
        "--dt-minutes", type=float, default=15.0, help="Time step in minutes"
    )
    parser.add_argument(
        "--label-id-column",
        default="MEDIAN_INTENSITY_CH3",
        help="Column name for label ID",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        # default=1,
        default=0.068030333725549 * 9.0,
        help="Factor by which image was downscaled (default: 9x downscale of original size)",
    )
    parser.add_argument(
        "--screenshots",
        "-s",
        nargs="+",
        type=int,
        help="Frame numbers to export as screenshots",
    )
    parser.add_argument(
        "--screenshot-dir",
        default=".",
        help="Output directory for screenshots",
    )
    parser.add_argument(
        "--screenshot-prefix",
        default="frame",
        help="Prefix for screenshot filenames",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Don't start the GUI (for batch processing)",
    )

    args = parser.parse_args()

    viewer = main(
        tracks_file=args.tracks,
        image_file=args.image,
        screenshot_frames=args.screenshots,
        screenshot_dir=args.screenshot_dir,
        screenshot_prefix=args.screenshot_prefix,
        cyan_channel_idx=args.cyan_channel,
        magenta_channel_idx=args.magenta_channel,
        label_channel_idx=args.labels_channel,
        dt_minutes=args.dt_minutes,
        label_id_column=args.label_id_column,
        scale_factor=args.scale_factor,
    )

    if not args.no_gui:
        napari.run()
