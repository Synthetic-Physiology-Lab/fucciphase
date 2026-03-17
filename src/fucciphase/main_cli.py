import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fucciphase import process_dataframe, process_trackmate
from fucciphase.phase import estimate_percentage_by_subsequence_alignment
from fucciphase.sensor import FUCCISASensor, get_fuccisa_default_sensor

logger = logging.getLogger(__name__)

TABULAR_SUFFIXES = {".csv", ".xlsx"}


def _read_table(path_str: str, description: str) -> pd.DataFrame:
    """Read a CSV or XLSX table with consistent error handling."""
    path = Path(path_str)
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".xlsx":
            return pd.read_excel(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} not found: {path_str}") from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"{description} is empty: {path_str}") from None
    except (pd.errors.ParserError, ValueError) as err:
        raise ValueError(f"Failed to parse {description} {path_str}: {err}") from None

    raise ValueError(f"{description} must be a CSV or XLSX file.")


def main_cli() -> None:
    """Command-line entry point for FUCCIphase."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="fucciphase",
        description="Estimate FUCCI cell-cycle phases and percentages.",
        epilog="Please report bugs and errors on GitHub.",
    )

    parser.add_argument(
        "tracking_file",
        type=str,
        help="TrackMate XML or tabular CSV/XLSX file",
    )
    parser.add_argument(
        "-ref",
        "--reference_file",
        type=str,
        help="Reference cell cycle CSV or XLSX file",
        required=True,
    )
    parser.add_argument(
        "--sensor_file",
        type=str,
        help=(
            "Sensor file in JSON format. If omitted, the default FUCCI SA sensor "
            "is used."
        ),
        default=None,
    )
    parser.add_argument(
        "-dt",
        "--timestep",
        type=float,
        help="Timestep in hours",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--magenta_channel",
        type=str,
        help="Name of the magenta channel in the input table",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cyan_channel",
        type=str,
        help="Name of the cyan channel in the input table",
        required=True,
    )
    parser.add_argument(
        "--generate_unique_tracks",
        action="store_true",
        help="Split TrackMate subtracks into unique tracks",
    )

    args = parser.parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    reference_df = _read_table(args.reference_file, "Reference file")
    reference_df.rename(
        columns={"cyan": args.cyan_channel, "magenta": args.magenta_channel},
        inplace=True,
    )

    if args.sensor_file is not None:
        try:
            with open(args.sensor_file) as fp:
                sensor_properties = json.load(fp)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Sensor file not found: {args.sensor_file}"
            ) from None
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in sensor file {args.sensor_file}: {e}"
            ) from None
        sensor = FUCCISASensor(**sensor_properties)
    else:
        sensor = get_fuccisa_default_sensor()

    tracking_path = Path(args.tracking_file)
    tracking_suffix = tracking_path.suffix.lower()

    if tracking_suffix == ".xml":
        df = process_trackmate(
            args.tracking_file,
            channels=[args.cyan_channel, args.magenta_channel],
            sensor=sensor,
            thresholds=[0.1, 0.1],
            generate_unique_tracks=args.generate_unique_tracks,
            output_dir=output_dir,
        )
    elif tracking_suffix in TABULAR_SUFFIXES:
        df = _read_table(args.tracking_file, "Tracking file")
        process_dataframe(
            df,
            channels=[args.cyan_channel, args.magenta_channel],
            sensor=sensor,
            thresholds=[0.1, 0.1],
            generate_unique_tracks=args.generate_unique_tracks,
        )
    else:
        raise ValueError("Tracking file must be an XML, CSV, or XLSX file.")

    track_id_name = "TRACK_ID"
    if args.generate_unique_tracks and "UNIQUE_TRACK_ID" in df.columns:
        track_id_name = "UNIQUE_TRACK_ID"

    estimate_percentage_by_subsequence_alignment(
        df,
        dt=args.timestep,
        channels=[args.cyan_channel, args.magenta_channel],
        reference_data=reference_df,
        track_id_name=track_id_name,
    )
    output_csv = output_dir / f"{tracking_path.stem}_processed.csv"
    df.to_csv(output_csv, index=False)
    logger.info("Wrote processed table to %s", output_csv)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    try:
        import napari
    except ImportError as err:
        raise ImportError("Install napari.") from err

    from fucciphase.napari import add_trackmate_data_to_viewer

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
    try:
        from aicsimageio import AICSImage  # type: ignore[attr-defined]

        AICSIMAGE = True
    except ImportError:
        try:
            import bioio_ome_tiff
            from bioio import BioImage
        except ImportError as err:
            raise ImportError(
                "Please install aicsimageio or bioio to read videos. "
                "Install with: pip install aicsimageio "
                "or pip install bioio bioio-ome-tiff"
            ) from err

    if AICSIMAGE:
        image = AICSImage(args.video)
    else:
        image = BioImage(args.video, reader=bioio_ome_tiff.Reader)

    # Determine spatial scale; fall back to unit scale or user-provided pixel size
    scale = (image.physical_pixel_sizes.Y, image.physical_pixel_sizes.X)
    if None in scale:
        if args.pixel_size is not None:
            scale = (args.pixel_size, args.pixel_size)
        else:
            logger.warning("No pixel sizes found in image metadata, using unit scale")
            scale = (1.0, 1.0)
    cyan = image.get_image_dask_data("TYX", C=args.cyan_channel)
    magenta = image.get_image_dask_data("TYX", C=args.magenta_channel)
    masks = image.get_image_dask_data("TYX", C=args.segmask_channel)

    try:
        track_df = pd.read_csv(args.fucciphase_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"FUCCIphase file not found: {args.fucciphase_file}"
        ) from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"FUCCIphase file is empty: {args.fucciphase_file}") from None
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Failed to parse FUCCIphase file {args.fucciphase_file}: {e}"
        ) from None

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
