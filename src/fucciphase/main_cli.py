import argparse
import json
from pathlib import Path

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
    """Fucciphase CLI: Command-line entry point for FUCCIphase.

    This function is invoked by the ``fucciphase`` console script and
    implements the standard command-line workflow:

    1. Parse command-line arguments describing:
       - a TrackMate tracking file (XML or CSV),
       - a reference cell-cycle trace in CSV format,
       - an optional FUCCI sensor JSON file,
       - the acquisition timestep and channel names.
    2. Load the reference data and rename its fluorescence columns to
       match the user-specified channel names.
    3. Load and preprocess the tracking data using either
       :func:`process_trackmate` (for XML) or
       :func:`process_dataframe` (for CSV).
    4. Estimate cell-cycle percentages for each track by subsequence
       alignment against the reference trace.
    5. Write the processed table to ``<tracking_file>_processed.csv`` in
       the same directory as the input file.

    The function is designed to be used from the command line and does
    not return a value. It will raise a ``ValueError`` if the tracking
    file does not have an XML or CSV extension.
    """
    parser = argparse.ArgumentParser(
        prog="fucciphase",
        description="FUCCIphase tool to estimate cell cycle phases and percentages.",
        epilog="Please report bugs and errors on GitHub.",
    )

    # -------------- 1. Parse command-line arguments --------------
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
    # Decide where to store outputs (CSV and, for XML input, processed XML)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # ---------------- 2. Load and adapt the reference cell-cycle trace ----------------
    reference_df = pd.read_csv(args.reference_file)
    # The reference file is expected to contain 'cyan' and 'magenta' columns;
    # they are renamed here to match the actual channel names in the data.
    reference_df.rename(
        columns={"cyan": args.cyan_channel, "magenta": args.magenta_channel},
        inplace=True,
    )
    # ---------------- 3. Build the sensor model ----------------
    if args.sensor_file is not None:
        with open(args.sensor_file) as fp:
            sensor_properties = json.load(fp)
        sensor = FUCCISASensor(**sensor_properties)
    else:
        sensor = get_fuccisa_default_sensor()

    # ---------------- 4. Load and preprocess the tracking data ----------------
    if args.tracking_file.endswith(".xml"):
        # XML: let process_trackmate handle I/O and preprocessing
        df = process_trackmate(
            args.tracking_file,
            channels=[args.cyan_channel, args.magenta_channel],
            sensor=sensor,
            thresholds=[0.1, 0.1],
            generate_unique_tracks=args.generate_unique_tracks,
            output_dir=output_dir,
        )
    elif args.tracking_file.endswith(".csv"):
        # CSV: read the table and then run the processing pipeline on it
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

    # ---------------- 5. Estimate cell-cycle percentages ----------------
    track_id_name = "UNIQUE_TRACK_ID"
    if not args.generate_unique_tracks:
        track_id_name = "TRACK_ID"

    estimate_percentage_by_subsequence_alignment(
        df,
        dt=args.timestep,
        channels=[args.cyan_channel, args.magenta_channel],
        reference_data=reference_df,
        track_id_name=track_id_name,
    )
    # ---------------- 6. Save results ----------------
    tracking_path = Path(args.tracking_file)
    output_csv = output_dir / (tracking_path.stem + "_processed.csv")
    df.to_csv(output_csv, index=False)


def main_visualization() -> None:
    """Fucciphase visualization.

    Launch a napari-based visualization of FUCCIphase results.

    This command-line entry point loads a processed FUCCIphase CSV file
    together with the corresponding OME-TIFF time-lapse movie and
    segmentation masks, then opens an interactive napari viewer showing:

    - cyan and magenta fluorescence channels,
    - segmentation masks as a labels layer,
    - tracks and cell-cycle information overlaid on the image.

    The function is intended to be invoked via the ``fucciphase-napari``
    console script and does not return a value.

    """
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
        default=None,
    )

    args = parser.parse_args()

    # Decide where to store outputs (CSV and, for XML input, processed XML)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Try to read the video using AICSImage; fall back to bioio if needed
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

    # Determine spatial scale; fall back to unit scale or user-provided pixel size
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
