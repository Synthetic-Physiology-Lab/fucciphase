import argparse
import json

import pandas as pd

from fucciphase import process_dataframe, process_trackmate
from fucciphase.phase import estimate_percentage_by_subsequence_alignment
from fucciphase.sensor import FUCCISASensor, get_fuccisa_default_sensor


def main_cli() -> None:
    """Fucciphase CLI."""
    parser = argparse.ArgumentParser(
        prog="fucciphase",
        description="FUCCIphase tool to estimate cell cycle phases and percentages.",
        epilog="Please report bugs and errors on GitHub.",
    )
    parser.add_argument("tracking_file", type=str, help="TrackMate XML or CSV file")
    parser.add_argument(
        "-ref", "--reference_file", type=str, help="Reference cell cycle CSV file", required=True
    )
    parser.add_argument(
        "--sensor_file",
        type=str,
        help="sensor file in JSON format "
        "(can be skipped, then FUCCI SA sensor is used by default)",
        default=None,
    )
    parser.add_argument("-dt", "--timestep", type=float, help="timestep in hours", required=True)
    parser.add_argument(
        "-m", "--magenta_channel", type=str, help="Name of magenta channel in TrackMate file", required=True
    )
    parser.add_argument(
        "-c", "--cyan_channel", type=str, help="Name of cyan channel in TrackMate file", required=True
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

    estimate_percentage_by_subsequence_alignment(
        df,
        dt=args.timestep,
        channels=[args.cyan_channel, args.magenta_channel],
        reference_data=reference_df,
        track_id_name="UNIQUE_TRACK_ID",
    )
    df.to_csv(args.tracking_file + "_processed.csv", index=False)

if __name__ == "__main__":
    main_cli()
