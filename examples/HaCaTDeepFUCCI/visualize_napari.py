"""
Visualize FUCCIphase results in Napari and create video.

This script loads processed track data from CSV, displays it in Napari
with image data, and optionally creates an animation or screenshots.

Usage:
    python visualize_napari.py --tracks processed_tracks.csv --image 2.nd2
    python visualize_napari.py --tracks processed_tracks.csv --image 2.nd2
                               --screenshots 0 13 26 39
"""

import argparse
import os

import napari
import numpy as np
import pandas as pd
import vispy.color.colormap
from skimage.io import imread

from fucciphase.napari import add_trackmate_data_to_viewer, pandas_df_to_napari_tracks

HASAICSIMAGE = True
try:
    from aicsimageio import AICSImage
except ImportError:
    import bioio_nd2
    from bioio import BioImage

    HASAICSIMAGE = False


def load_image_data(
    image_file: str,
    cyan_channel_idx: int = 3,
    magenta_channel_idx: int = 0,
    tubulin_channel_idx: int | None = 2,
) -> dict:
    """Load image data from ND2 file.

    Parameters
    ----------
    image_file : str
        Path to image file (ND2 format)
    cyan_channel_idx : int
        Channel index for cyan
    magenta_channel_idx : int
        Channel index for magenta
    tubulin_channel_idx : int or None
        Channel index for tubulin, None to skip

    Returns
    -------
    dict
        Dictionary with image data and metadata
    """
    if HASAICSIMAGE:
        image = AICSImage(image_file)
    else:
        image = BioImage(image_file, reader=bioio_nd2.Reader)
    scale = (image.physical_pixel_sizes.Y, image.physical_pixel_sizes.X)

    cyan = image.get_image_data("TYX", C=cyan_channel_idx)
    magenta = image.get_image_data("TYX", C=magenta_channel_idx)

    result = {
        "scale": scale,
        "cyan": cyan,
        "magenta": magenta,
        "cyan_contrast": (np.percentile(cyan, 1), np.percentile(cyan, 99.9)),
        "magenta_contrast": (np.percentile(magenta, 1), np.percentile(magenta, 99.9)),
    }

    if tubulin_channel_idx is not None:
        tubulin = image.get_image_data("TYX", C=tubulin_channel_idx)
        result["tubulin"] = tubulin
        result["tubulin_contrast"] = (
            np.percentile(tubulin, 1),
            np.percentile(tubulin, 99.9),
        )

    return result


def crop_images(
    image_data: dict,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> dict:
    """Crop images to a spatial region.

    Parameters
    ----------
    image_data : dict
        Dictionary with image arrays and scale
    x_limits : tuple
        (min_x, max_x) in physical units
    y_limits : tuple
        (min_y, max_y) in physical units

    Returns
    -------
    dict
        Dictionary with cropped images
    """
    scale = image_data["scale"]
    x_slice = slice(int(x_limits[0] / scale[0]), int(x_limits[1] / scale[0]))
    y_slice = slice(int(y_limits[0] / scale[1]), int(y_limits[1] / scale[1]))

    result = {"scale": scale}

    for key in ["cyan", "magenta", "tubulin"]:
        if key in image_data:
            result[key] = image_data[key][..., y_slice, x_slice]

    for key in ["cyan_contrast", "magenta_contrast", "tubulin_contrast"]:
        if key in image_data:
            result[key] = image_data[key]

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
    """Export screenshots at specific frames.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    frames : list of int
        Frame numbers to screenshot
    output_dir : str
        Output directory for screenshots
    prefix : str
        Prefix for output filenames
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for frame in frames:
        # Ensure overlays are visible
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 64
        viewer.scale_bar.visible = True
        viewer.scale_bar.font_size = 0

        # Set the frame
        viewer.dims.current_step = (frame, 0, 0)

        # Export screenshot
        output_path = os.path.join(output_dir, f"{prefix}_{frame}.png")
        viewer.export_figure(output_path)
        print(f"Saved screenshot: {output_path}")


def create_animation(
    viewer: napari.Viewer,
    output_file: str,
    n_frames: int,
    fps: int = 4,
    quality: int = 5,
) -> None:
    """Create animation from viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    output_file : str
        Output video file path
    n_frames : int
        Number of frames in the animation
    fps : int
        Frames per second
    quality : int
        Video quality (1-10)
    """
    from napari_animation import Animation

    animation = Animation(viewer)

    viewer.text_overlay.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.font_size = 0

    # Start on first frame
    viewer.dims.current_step = (0, 0, 0)
    animation.capture_keyframe()

    # End on last frame
    viewer.dims.current_step = (n_frames - 1, 0, 0)
    animation.capture_keyframe(steps=n_frames - 1)

    print(f"Rendering animation to {output_file}...")
    animation.animate(
        output_file, canvas_only=True, fps=fps, quality=quality, scale_factor=1.0
    )
    print("Done!")


# ruff: noqa: C901
def main(
    tracks_file: str,
    image_file: str | None = None,
    labels_file: str | None = None,
    output_video: str | None = None,
    screenshot_frames: list[int] | None = None,
    screenshot_dir: str = ".",
    screenshot_prefix: str = "frame",
    cyan_channel_idx: int = 3,
    magenta_channel_idx: int = 0,
    tubulin_channel_idx: int | None = 2,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    dt_minutes: float = 15.0,
    label_id_column: str = "MEDIAN_INTENSITY_CH3",
    fps: int = 4,
) -> napari.Viewer:
    """Visualize tracks in Napari.

    Parameters
    ----------
    tracks_file : str
        Path to processed tracks CSV
    image_file : str or None
        Path to image file (ND2 format)
    labels_file : str or None
        Path to labels TIF file
    output_video : str or None
        Path for output video, None to skip
    screenshot_frames : list of int or None
        Frame numbers to export as screenshots, None to skip
    screenshot_dir : str
        Output directory for screenshots
    screenshot_prefix : str
        Prefix for screenshot filenames
    cyan_channel_idx : int
        Channel index for cyan
    magenta_channel_idx : int
        Channel index for magenta
    tubulin_channel_idx : int or None
        Channel index for tubulin, None to skip
    x_limits : tuple or None
        (min_x, max_x) for cropping
    y_limits : tuple or None
        (min_y, max_y) for cropping
    dt_minutes : float
        Time step in minutes
    label_id_column : str
        Column name for label ID
    fps : int
        Frames per second for video

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

    # Load image data if provided
    image_data = None
    if image_file:
        print(f"Loading image from {image_file}...")
        image_data = load_image_data(
            image_file,
            cyan_channel_idx=cyan_channel_idx,
            magenta_channel_idx=magenta_channel_idx,
            tubulin_channel_idx=tubulin_channel_idx,
        )

        if x_limits is not None and y_limits is not None:
            image_data = crop_images(image_data, x_limits, y_limits)

    # Load labels if provided
    labels = None
    if labels_file:
        print(f"Loading labels from {labels_file}...")
        labels = imread(labels_file)
        if x_limits is not None and y_limits is not None and image_data:
            scale = image_data["scale"]
            x_slice = slice(int(x_limits[0] / scale[0]), int(x_limits[1] / scale[0]))
            y_slice = slice(int(y_limits[0] / scale[1]), int(y_limits[1] / scale[1]))
            labels = labels[..., y_slice, x_slice]

    # Create viewer
    print("Creating Napari viewer...")
    viewer = napari.Viewer()
    setup_viewer(viewer, dt_minutes=dt_minutes)

    scale = image_data["scale"] if image_data else (1.0, 1.0)

    # Add trackmate data
    if image_data:
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

        # Add tubulin layer if available
        if "tubulin" in image_data:
            tubulin_layer = viewer.add_image(
                image_data["tubulin"],
                colormap="gray",
                blending="additive",
                scale=scale,
            )
            tubulin_layer.contrast_limits = image_data["tubulin_contrast"]
            tubulin_layer.visible = False

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

    # Create animation if requested
    if output_video:
        n_frames = (
            labels.shape[0] if labels is not None else track_df["FRAME"].max() + 1
        )
        create_animation(viewer, output_video, n_frames, fps=fps)

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
        description="Visualize FUCCIphase results in Napari"
    )
    parser.add_argument(
        "--tracks", "-t", required=True, help="Processed tracks CSV file"
    )
    parser.add_argument("--image", "-i", help="Image file (ND2 format)")
    parser.add_argument("--labels", "-l", help="Labels TIF file")
    parser.add_argument("--output", "-o", help="Output video file")
    parser.add_argument(
        "--cyan-channel", type=int, default=3, help="Cyan channel index"
    )
    parser.add_argument(
        "--magenta-channel", type=int, default=0, help="Magenta channel index"
    )
    parser.add_argument(
        "--tubulin-channel",
        type=int,
        default=None,
        help="Tubulin channel index (optional)",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Crop region (x_min x_max y_min y_max)",
    )
    parser.add_argument(
        "--dt-minutes", type=float, default=15.0, help="Time step in minutes"
    )
    parser.add_argument(
        "--label-id-column",
        default="MEDIAN_INTENSITY_CH3",
        help="Column name for label ID",
    )
    parser.add_argument("--fps", type=int, default=4, help="Video frames per second")
    parser.add_argument(
        "--screenshots",
        "-s",
        nargs="+",
        type=int,
        help="Frame numbers to export as screenshots (e.g., --screenshots 0 13 26 39)",
    )
    parser.add_argument(
        "--screenshot-dir",
        default=".",
        help="Output directory for screenshots (default: current directory)",
    )
    parser.add_argument(
        "--screenshot-prefix",
        default="frame",
        help="Prefix for screenshot filenames (default: 'frame')",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Don't start the GUI (for batch processing)",
    )

    args = parser.parse_args()

    x_limits = None
    y_limits = None
    if args.crop:
        x_limits = (args.crop[0], args.crop[1])
        y_limits = (args.crop[2], args.crop[3])

    viewer = main(
        tracks_file=args.tracks,
        image_file=args.image,
        labels_file=args.labels,
        output_video=args.output,
        screenshot_frames=args.screenshots,
        screenshot_dir=args.screenshot_dir,
        screenshot_prefix=args.screenshot_prefix,
        cyan_channel_idx=args.cyan_channel,
        magenta_channel_idx=args.magenta_channel,
        tubulin_channel_idx=args.tubulin_channel,
        x_limits=x_limits,
        y_limits=y_limits,
        dt_minutes=args.dt_minutes,
        label_id_column=args.label_id_column,
        fps=args.fps,
    )

    if not args.no_gui:
        napari.run()
