"""
Compute cell cycle percentages using FUCCIphase and export to CSV.

This script processes TrackMate tracking data, estimates cell cycle phases
and percentages using DTW alignment, filters tracks, and exports the results.

Usage:
    python compute_fucciphase.py
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fucciphase import process_trackmate
from fucciphase.phase import SignalMode, estimate_percentage_by_subsequence_alignment
from fucciphase.sensor import FUCCISASensor
from fucciphase.utils import postprocess_estimated_percentages


def load_sensor(sensor_file: str) -> FUCCISASensor:
    """Load sensor configuration from JSON file."""
    with open(sensor_file) as fp:
        sensor_properties = json.load(fp)
    return FUCCISASensor(**sensor_properties)


def filter_tracks(
    track_df: pd.DataFrame,
    minimum_length: int = 12,
    dtw_threshold: float = np.inf,
    track_id_name: str = "TRACK_ID",
) -> pd.DataFrame:
    """Filter tracks based on length and DTW distortion.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with track data including CELL_CYCLE_PERC_DTW and DTW_DISTORTION_REL
    minimum_length : int
        Minimum number of frames a track must have
    dtw_threshold : float
        Maximum DTW distortion threshold for filtering
    track_id_name : str
        Name of the track ID column

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    sub_track_nona = track_df[track_df["CELL_CYCLE_PERC_DTW"].notna()].copy()

    # Get unique tracks with sufficient length
    unique_tracks_tmp = track_df[track_id_name].unique()
    unique_tracks = []
    for unique_track in unique_tracks_tmp:
        if unique_track < 0:
            continue
        if (
            len(sub_track_nona[sub_track_nona[track_id_name] == unique_track])
            < minimum_length
        ):
            continue
        unique_tracks.append(unique_track)
    unique_tracks = np.sort(unique_tracks)

    # Filter by DTW distortion
    sub_df = track_df[[track_id_name, "DTW_DISTORTION_REL"]].drop_duplicates()
    sub_df = sub_df[sub_df[track_id_name].isin(unique_tracks)]

    filter_ids = []
    for _, row in sub_df.iterrows():
        if row["DTW_DISTORTION_REL"] < dtw_threshold:
            track_length = len(track_df[track_df[track_id_name] == row[track_id_name]])
            if track_length >= minimum_length:
                filter_ids.append(row[track_id_name])
    filter_ids = np.sort(filter_ids)

    return track_df[track_df[track_id_name].isin(filter_ids)].copy()


def crop_tracks(
    track_df: pd.DataFrame,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> pd.DataFrame:
    """Crop tracks to a spatial region.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with POSITION_X and POSITION_Y columns
    x_limits : tuple
        (min_x, max_x) limits
    y_limits : tuple
        (min_y, max_y) limits

    Returns
    -------
    pd.DataFrame
        Cropped DataFrame with adjusted positions
    """
    crop_df = track_df[
        (track_df["POSITION_X"] > x_limits[0])
        & (track_df["POSITION_X"] < x_limits[1])
        & (track_df["POSITION_Y"] > y_limits[0])
        & (track_df["POSITION_Y"] < y_limits[1])
    ].copy()

    crop_df["POSITION_X"] -= x_limits[0]
    crop_df["POSITION_Y"] -= y_limits[0]

    return crop_df


def _normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalize data to [0, 1] range."""
    dmin, dmax = data.min(), data.max()
    if dmax - dmin == 0:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def plot_reference_vs_query(
    track_df: pd.DataFrame,
    reference_track: pd.DataFrame,
    channels: list[str],
    output_dir: str = "figures",
    dtw_threshold: float = np.inf,
    track_id_name: str = "TRACK_ID",
    minimum_length: int = 10,
    dt: float = 0.25,
) -> None:
    """Plot reference vs query curves for each track.

    Generates comparison plots showing how each track aligns to the reference
    curve. Plots are saved as PDF files named good_ID_<id>.pdf or bad_ID_<id>.pdf
    based on whether the DTW distortion is below or above the threshold.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with track data including CELL_CYCLE_PERC_DTW and DTW_DISTORTION_REL
    reference_track : pd.DataFrame
        Reference track with time, percentage, and channel columns
    channels : list[str]
        List of channel column names (e.g., ["MEAN_INTENSITY_CH2", ...])
    output_dir : str
        Directory to save plots
    dtw_threshold : float
        Threshold for classifying tracks as good/bad. Default is np.inf (all good).
    track_id_name : str
        Name of the track ID column
    minimum_length : int
        Minimum track length to plot
    dt : float
        Time step in hours
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique tracks with valid DTW results
    sub_df = track_df[[track_id_name, "DTW_DISTORTION_REL"]].drop_duplicates()
    sub_df = sub_df[sub_df["DTW_DISTORTION_REL"].notna()]

    cyan_channel = channels[0]
    magenta_channel = channels[1]

    for _, row in sub_df.iterrows():
        track_id = row[track_id_name]
        distortion = row["DTW_DISTORTION_REL"]

        t_df = track_df[track_df[track_id_name] == track_id]

        if len(t_df) < minimum_length:
            continue

        # Determine good/bad based on threshold
        prefix = "good" if distortion < dtw_threshold else "bad"

        # Create figure with 2 rows (cyan, magenta) x 2 cols (time, warped)
        fig, axes = plt.subplots(2, 2, sharex="col", figsize=(10, 6))
        ax1, ax2 = axes[0], axes[1]

        fig.suptitle(f"Track ID {track_id}, DTW Distortion: {distortion:.2f}")

        ax1[0].set_title("Time")
        ax1[1].set_title("Warped (percentage)")

        # Time axis plots
        if "time" in t_df.columns:
            track_time = t_df["time"] - t_df["time"].min()
        else:
            track_time = dt * (t_df["FRAME"] - t_df["FRAME"].min())

        # Cyan channel
        ax1[0].plot(
            reference_track["time"],
            reference_track[cyan_channel],
            color="cyan",
            lw=4,
            label="Reference",
        )
        ax1[0].plot(
            track_time,
            _normalize(t_df[cyan_channel].values),
            color="blue",
            lw=2,
            label="Query",
        )

        # Magenta channel
        ax2[0].plot(
            reference_track["time"],
            reference_track[magenta_channel],
            color="magenta",
            lw=4,
        )
        ax2[0].plot(
            track_time,
            _normalize(t_df[magenta_channel].values),
            color="blue",
            lw=2,
        )

        # Warped (percentage) plots
        ax1[1].plot(
            reference_track["percentage"],
            reference_track[cyan_channel],
            color="cyan",
            lw=4,
        )
        ax1[1].plot(
            t_df["CELL_CYCLE_PERC_DTW"],
            _normalize(t_df[cyan_channel].values),
            color="blue",
            lw=2,
        )

        ax2[1].plot(
            reference_track["percentage"],
            reference_track[magenta_channel],
            color="magenta",
            lw=4,
        )
        ax2[1].plot(
            t_df["CELL_CYCLE_PERC_DTW"],
            _normalize(t_df[magenta_channel].values),
            color="blue",
            lw=2,
        )

        # Labels and formatting
        ax1[0].set_ylabel("Intensity (norm.)")
        ax2[0].set_ylabel("Intensity (norm.)")
        ax2[0].set_xlabel("Time / h")
        ax2[1].set_xlabel("Percentage")

        ax1[1].set_xlim(0, 100)
        ax2[1].set_xlim(0, 100)

        for ax in [ax1[0], ax1[1], ax2[0], ax2[1]]:
            ax.set_yticks([])

        ax1[0].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_ID_{track_id}.pdf"))
        plt.close(fig)


def main(
    track_file: str = "merged.ome.xml",
    sensor_file: str = "fuccisa_hacat.json",
    reference_file: str = "hacat_fucciphase_reference.csv",
    output_file: str = "processed_tracks.csv",
    cyan_channel: str = "MEAN_INTENSITY_CH2",
    magenta_channel: str = "MEAN_INTENSITY_CH1",
    dt: float = 0.25,
    minimum_length: int = 12,
    dtw_threshold: float = np.inf,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    signal_mode: SignalMode = "derivative",
    penalty: float = 0.05,
    signal_weight: float = 1.0,
    signal_smooth: int = 0,
    plot_figures: bool = False,
    figures_dir: str = "figures",
) -> pd.DataFrame:
    """Process TrackMate data and export to CSV.

    Parameters
    ----------
    track_file : str
        Path to TrackMate XML file
    sensor_file : str
        Path to sensor configuration JSON file
    reference_file : str
        Path to reference track CSV file
    output_file : str
        Path for output CSV file
    cyan_channel : str
        Column name for cyan channel intensity
    magenta_channel : str
        Column name for magenta channel intensity
    dt : float
        Time step in hours
    minimum_length : int
        Minimum track length in frames
    dtw_threshold : float
        Maximum DTW distortion for filtering. Default is np.inf (no filtering).
    x_limits : tuple or None
        (min_x, max_x) for spatial cropping, None to skip
    y_limits : tuple or None
        (min_y, max_y) for spatial cropping, None to skip
    signal_mode : SignalMode
        DTW signal processing mode: "signal", "derivative", or "both"
    penalty : float
        Penalty for DTW algorithm, enforces diagonal warping path
    signal_weight : float
        Weight for signal relative to derivative in "both" mode.
        Default 1.0 means equal contribution. Values > 1.0 weight signal higher.
    signal_smooth : int
        Window size for signal smoothing (Savitzky-Golay, polyorder=3).
        0 means no smoothing, must be > 3 if used.
    plot_figures : bool
        If True, generate reference vs query comparison plots for each track
    figures_dir : str
        Directory to save comparison plots

    Returns
    -------
    pd.DataFrame
        Processed track DataFrame
    """
    print(f"Loading sensor from {sensor_file}...")
    sensor = load_sensor(sensor_file)

    print(f"Processing TrackMate file {track_file}...")
    track_df = process_trackmate(
        track_file,
        channels=[cyan_channel, magenta_channel],
        sensor=sensor,
        thresholds=[0.1, 0.1],
        use_moving_average=True,
        window_size=10,
    )

    print(f"Loading reference track from {reference_file}...")
    reference_track = pd.read_csv(reference_file)
    reference_track.rename(
        columns={"cyan": cyan_channel, "magenta": magenta_channel}, inplace=True
    )
    reference_track[cyan_channel + "_NORM"] = reference_track[cyan_channel]
    reference_track[magenta_channel + "_NORM"] = reference_track[magenta_channel]

    print(f"Estimating cell cycle percentages by DTW alignment (mode={signal_mode})...")
    estimate_percentage_by_subsequence_alignment(
        track_df,
        dt=dt,
        channels=[cyan_channel + "_NORM", magenta_channel + "_NORM"],
        reference_data=reference_track,
        track_id_name="TRACK_ID",
        signal_mode=signal_mode,
        penalty=penalty,
        signal_weight=signal_weight,
        signal_smooth=signal_smooth,
    )
    postprocess_estimated_percentages(
        track_df, "CELL_CYCLE_PERC_DTW", track_id_name="TRACK_ID"
    )

    # Add time column
    track_df["time"] = dt * track_df["FRAME"]

    # Add normalized percentage for visualization
    track_df["CELL_CYCLE_PERC_NORM"] = track_df["CELL_CYCLE_PERC_DTW"] / 100.0

    # Plot reference vs query curves if requested
    if plot_figures:
        print(f"Generating comparison plots in {figures_dir}/...")
        plot_reference_vs_query(
            track_df,
            reference_track,
            channels=[cyan_channel, magenta_channel],
            output_dir=figures_dir,
            dtw_threshold=dtw_threshold,
            track_id_name="TRACK_ID",
            minimum_length=minimum_length,
            dt=dt,
        )

    print(
        "Filtering tracks "
        f"(min_length={minimum_length}, dtw_threshold={dtw_threshold})..."
    )
    track_df = filter_tracks(
        track_df,
        minimum_length=minimum_length,
        dtw_threshold=dtw_threshold,
    )

    if x_limits is not None and y_limits is not None:
        print(f"Cropping to region x={x_limits}, y={y_limits}...")
        track_df = crop_tracks(track_df, x_limits, y_limits)

    print(f"Saving to {output_file}...")
    track_df.to_csv(output_file, index=False)

    n_tracks = track_df["TRACK_ID"].nunique()
    n_points = len(track_df)
    print(f"Done! Exported {n_points} data points from {n_tracks} tracks.")

    return track_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cell cycle percentages using FUCCIphase"
    )
    parser.add_argument(
        "--track-file", default="merged.ome.xml", help="TrackMate XML file"
    )
    parser.add_argument(
        "--sensor-file", default="fuccisa_hacat.json", help="Sensor configuration JSON"
    )
    parser.add_argument(
        "--reference-file",
        default="hacat_fucciphase_reference.csv",
        help="Reference track CSV",
    )
    parser.add_argument(
        "--output", "-o", default="processed_tracks.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--cyan-channel", default="MEAN_INTENSITY_CH2", help="Cyan channel column name"
    )
    parser.add_argument(
        "--magenta-channel",
        default="MEAN_INTENSITY_CH1",
        help="Magenta channel column name",
    )
    parser.add_argument("--dt", type=float, default=0.25, help="Time step in hours")
    parser.add_argument(
        "--min-length", type=int, default=12, help="Minimum track length"
    )
    parser.add_argument(
        "--dtw-threshold",
        type=float,
        default=float("inf"),
        help="DTW distortion threshold for filtering (default: inf, no filtering)",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Crop region (x_min x_max y_min y_max)",
    )
    parser.add_argument(
        "--signal-mode",
        choices=["signal", "derivative", "both"],
        default="derivative",
        help="DTW signal processing mode (default: derivative)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.05,
        help="DTW penalty to enforce diagonal warping path (default: 0.05)",
    )
    parser.add_argument(
        "--signal-weight",
        type=float,
        default=1.0,
        help="Weight for signal vs derivative in 'both' mode "
        "(default: 1.0, >1 weights signal higher)",
    )
    parser.add_argument(
        "--signal-smooth",
        type=int,
        default=0,
        help="Window size for signal smoothing (Savitzky-Golay filter). "
        "0 = no smoothing, must be > 3 if used (default: 0)",
    )
    parser.add_argument(
        "--plot-figures",
        action="store_true",
        help="Generate reference vs query comparison plots for each track",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for comparison plots (default: figures)",
    )

    args = parser.parse_args()

    x_limits = None
    y_limits = None
    if args.crop:
        x_limits = (args.crop[0], args.crop[1])
        y_limits = (args.crop[2], args.crop[3])

    main(
        track_file=args.track_file,
        sensor_file=args.sensor_file,
        reference_file=args.reference_file,
        output_file=args.output,
        cyan_channel=args.cyan_channel,
        magenta_channel=args.magenta_channel,
        dt=args.dt,
        minimum_length=args.min_length,
        dtw_threshold=args.dtw_threshold,
        x_limits=x_limits,
        y_limits=y_limits,
        signal_mode=args.signal_mode,
        penalty=args.penalty,
        signal_weight=args.signal_weight,
        signal_smooth=args.signal_smooth,
        plot_figures=args.plot_figures,
        figures_dir=args.figures_dir,
    )
