import argparse
import json

import pandas as pd

from fucciphase import process_dataframe, process_trackmate
from fucciphase.napari import add_trackmate_data_to_viewer
from fucciphase.phase import estimate_percentage_by_subsequence_alignment
from fucciphase.sensor import FUCCISASensor, get_fuccisa_default_sensor

try:
    import napari
except ImportError as err:
    raise ImportError("Install napari.") from err


def main_cli() -> None:
    """Fucciphase CLI."""
    parser = argparse.ArgumentParser(
        prog="fucciphase",
        description="FUCCIphase tool to estimate cell cycle phases and percentages.",
        epilog="Please report bugs and errors on GitHub.",
    )
    parser.add_argument("tracking_file", type=str, help="TrackMate XML or CSV file")
    parser.add_argument(
        "-ref",
        "--reference_file",
        type=str,
        help="Reference cell cycle CSV file",
        required=True,
    )
    parser.add_argument(
        "--sensor_file",
        type=str,
        help="sensor file in JSON format "
        "(can be skipped, then FUCCI SA sensor is used by default)",
        default=None,
    )
    parser.add_argument(
        "-dt", "--timestep", type=float, help="timestep in hours", required=True
    )
    parser.add_argument(
        "-m",
        "--magenta_channel",
        type=str,
        help="Name of magenta channel in TrackMate file",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cyan_channel",
        type=str,
        help="Name of cyan channel in TrackMate file",
        required=True,
    )
    parser.add_argument(
        "--generate_unique_tracks",
        type=bool,
        help="Split subtracks (TrackMate specific)",
        default=False,
    )

    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference_file)
    reference_df.rename(
        columns={"cyan": args.cyan_channel, "magenta": args.magenta_channel},
        inplace=True,
    )

    if args.sensor_file is not None:
        with open(args.sensor_file) as fp:
            sensor_properties = json.load(fp)
        sensor = FUCCISASensor(**sensor_properties)
    else:
        sensor = get_fuccisa_default_sensor()

    if args.tracking_file.endswith(".xml"):
        df = process_trackmate(
            args.tracking_file,
            channels=[args.cyan_channel, args.magenta_channel],
            sensor=sensor,
            thresholds=[0.1, 0.1],
            generate_unique_tracks=args.generate_unique_tracks,
        )
    elif args.tracking_file.endswith(".csv"):
        df = pd.read_csv(args.tracking_file)
        process_dataframe(
            df,
            channels=[args.cyan_channel, args.magenta_channel],
            sensor=sensor,
            thresholds=[0.1, 0.1],
            generate_unique_tracks=args.generate_unique_tracks,
        )
    else:
        raise ValueError("Tracking file must be an XML or CSV file.")

    track_id_name = "UNIQUE_TRACK_ID"
    if not args.generate_unique_tracks:
        track_id_name = "TRACK_ID"

    estimate_percentage_by_subsequence_alignment(
        df,
        dt=args.timestep,
        channels=[args.cyan_channel, args.magenta_channel],
        reference_data=reference_df,
        track_id_name=track_id_name
    )
    df.to_csv(args.tracking_file + "_processed.csv", index=False)


def main_visualization() -> None:
    """Fucciphase visualization."""
    parser = argparse.ArgumentParser(
        prog="fucciphase-napari",
        description="FUCCIphase napari script to launch visualization.",
        epilog="Please report bugs and errors on GitHub.",
    )
    parser.add_argument("fucciphase_file", type=str, help="Processed file.")
    parser.add_argument(
        "video", type=str, help="OME-Tiff file with video data and segmentation masks"
    )
    parser.add_argument(
        "-m",
        "--magenta_channel",
        type=int,
        help="Index of magenta channel in video file",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cyan_channel",
        type=int,
        help="Index of cyan channel in video file",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--segmask_channel",
        type=int,
        help="Index of segmentation mask channel in video file",
        required=True,
    )
    parser.add_argument(
            "--pixel_size",
            type=float,
            help="Pixel size, only used if not in metadata",
            default=None)

    args = parser.parse_args()

    AICSIMAGE = False
    BIOIMAGE = False
    try:
        from aicsimageio import AICSImage

        AICSIMAGE = True
    except ImportError as err:
        from bioio import BioImage

        BIOIMAGE = True
        import bioio_ome_tiff

        if not BIOIMAGE:
            raise ImportError(
                "Please install AICSImage or bioio to read videos"
            ) from err
    if AICSIMAGE:
        image = AICSImage(args.video)
    elif BIOIMAGE:
        image = BioImage(args.video, reader=bioio_ome_tiff.Reader)
    scale = (image.physical_pixel_sizes.Y, image.physical_pixel_sizes.X)
    if None in scale:
        if args.pixel_size is not None:
            scale = (args.pixel_size, args.pixel_size)
        else:
            print("WARNING: No pixel sizes found, using unit scale")
            scale = (1.0, 1.0)
    cyan = image.get_image_dask_data("TYX", C=args.cyan_channel)
    magenta = image.get_image_dask_data("TYX", C=args.magenta_channel)
    masks = image.get_image_dask_data("TYX", C=args.segmask_channel)
    track_df = pd.read_csv(args.fucciphase_file)

    viewer = napari.Viewer()

    add_trackmate_data_to_viewer(
        track_df,
        viewer,
        scale=scale,
        image_data=[cyan, magenta],
        colormaps=["cyan", "magenta"],
        labels=masks,
        cycle_percentage_id="CELL_CYCLE_PERC_DTW",
        textkwargs={"size": 14},
    )
    napari.run()


if __name__ == "__main__":
    main_cli()
