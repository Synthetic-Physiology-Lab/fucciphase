from itertools import cycle
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from scipy import interpolate

from .phase import NewColumns
from .utils import get_norm_channel_name


def set_phase_colors(
    df: pd.DataFrame, colordict: dict, phase_column: str = "DISCRETE_PHASE_MAX"
) -> None:
    """Label each phase by fixed color."""
    phases = df[phase_column].unique()
    if not all(phase in colordict for phase in phases):
        raise ValueError(f"Provide a color for every phase in: {phases}")

    df["COLOR"] = df[phase_column].copy()
    for phase in phases:
        df.loc[df[phase_column] == phase, "COLOR"] = colordict[phase]


def plot_feature(
    df: pd.DataFrame,
    time_column: str,
    feature_name: str,
    interpolate_time: bool = False,
    track_id_name: str = "TRACK_ID",
    ylim: Optional[tuple] = None,
    yticks: Optional[list] = None,
) -> Figure:
    """Plot features of individual tracks in one plot."""
    if feature_name not in df:
        raise ValueError(f"(Feature {feature_name} not in provided DataFrame.")
    if time_column not in df:
        raise ValueError(f"(Time {time_column} not in provided DataFrame.")
    tracks = df[track_id_name].unique()
    tracks = tracks[tracks >= 0]

    fig = plt.figure()
    # Plot each graph, and manually set the y tick values
    for track_idx in tracks:
        time = df.loc[df[track_id_name] == track_idx, time_column].to_numpy()
        feature = df.loc[df[track_id_name] == track_idx, feature_name].to_numpy()
        plt.plot(time, feature)
        if ylim is not None:
            plt.ylim(ylim)
        if yticks is not None:
            plt.yticks(yticks)
    return fig


# flake8: noqa: C901
def plot_feature_stacked(
    df: pd.DataFrame,
    time_column: str,
    feature_name: str,
    interpolate_time: bool = False,
    track_id_name: str = "TRACK_ID",
    ylim: Optional[tuple] = None,
    yticks: Optional[list] = None,
    interpolation_steps: int = 1000,
    figsize: Optional[tuple] = None,
    selected_tracks: Optional[List[int]] = None,
) -> Figure:
    """Stack features of individual tracks.


    Notes
    -----
    If `selected_tracks` are chosen, the averaging
    is still performed on all tracks.
    Few selected tracks are stacked to enhance visibility.
    """
    if feature_name not in df:
        raise ValueError(f"(Feature {feature_name} not in provided DataFrame.")
    if time_column not in df:
        raise ValueError(f"(Time {time_column} not in provided DataFrame.")
    if "COLOR" not in df:
        raise ValueError("Run set_phase_colors first on DataFrame")
    tracks = df[track_id_name].unique()
    tracks = tracks[tracks >= 0]
    if selected_tracks is None:
        selected_tracks = tracks
    else:
        if not set(selected_tracks).issubset(tracks):
            raise ValueError(
                "Selected tracks contain tracks that are not in track list."
            )
    if figsize is None:
        figsize = (10, 2 * len(selected_tracks))
    if not interpolate_time:
        fig, axs = plt.subplots(len(selected_tracks), 1, sharex=True, figsize=figsize)
    else:
        fig, axs = plt.subplots(
            len(selected_tracks) + 1, 1, sharex=True, figsize=figsize
        )
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    max_frame = 0
    min_frame = np.inf

    # Plot each graph, and manually set the y tick values
    for i, track_idx in enumerate(selected_tracks):
        time = df.loc[df[track_id_name] == track_idx, time_column].to_numpy()
        feature = df.loc[df[track_id_name] == track_idx, feature_name].to_numpy()
        colors = df.loc[df[track_id_name] == track_idx, "COLOR"].to_numpy()
        axs[i].plot(time, feature)
        axs[i].scatter(time, feature, c=colors, lw=4)
        if ylim is not None:
            axs[i].set_ylim(ylim)
        if yticks is not None:
            axs[i].set_yticks(yticks)
        if time.max() > max_frame:
            max_frame = time.max()
        if time.min() < min_frame:
            min_frame = time.min()

    if interpolate_time:
        interpolated_time = np.linspace(min_frame, max_frame, num=interpolation_steps)
        interpolated_feature = np.zeros(shape=(len(interpolated_time), len(tracks)))
        for i, track_idx in enumerate(tracks):
            time = df.loc[df[track_id_name] == track_idx, time_column].to_numpy()
            feature = df.loc[df[track_id_name] == track_idx, feature_name].to_numpy()
            interpolated_feature[:, i] = np.interp(
                interpolated_time, time, feature, left=np.nan, right=np.nan
            )
        axs[-1].plot(
            interpolated_time,
            np.nanmean(interpolated_feature, axis=1),
            lw=5,
            color="black",
        )
        if ylim is not None:
            axs[-1].set_ylim(ylim)
        if yticks is not None:
            axs[-1].set_yticks(yticks)

    return fig


def plot_raw_intensities(
    df: pd.DataFrame,
    channel1: str,
    channel2: str,
    color1: str = "cyan",
    color2: str = "magenta",
    time_column: str = "FRAME",
    time_label: str = "Frame #",
    **plot_kwargs: bool,
) -> None:
    """Plot intensities of two-channel sensor."""
    ch1_intensity = df[channel1]
    ch2_intensity = df[channel2]

    t = df[time_column]

    fig, ax1 = plt.subplots()

    # prepare axes
    ax1.set_xlabel(time_label)
    ax1.set_ylabel(channel1, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(channel2, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # plot signal
    ax1.plot(t, ch1_intensity, color=color1, **plot_kwargs)
    ax2.plot(t, ch2_intensity, color=color2, **plot_kwargs)
    fig.tight_layout()


def plot_normalized_intensities(
    df: pd.DataFrame,
    channel1: str,
    channel2: str,
    color1: str = "cyan",
    color2: str = "magenta",
    time_column: str = "FRAME",
    time_label: str = "Frame #",
    **plot_kwargs: bool,
) -> None:
    """Plot normalised intensities of two-channel sensor."""
    ch1_intensity = df[get_norm_channel_name(channel1)]
    ch2_intensity = df[get_norm_channel_name(channel2)]

    t = df[time_column]
    plt.plot(t, ch1_intensity, color=color1, label=channel1, **plot_kwargs)
    plt.plot(t, ch2_intensity, color=color2, label=channel2, **plot_kwargs)
    plt.xlabel(time_label)
    plt.ylabel("Normalised intensity")


def plot_phase(df: pd.DataFrame, channel1: str, channel2: str) -> None:
    """Plot the two channels and vertical lines
    corresponding to the change of phase.

    The dataframe must be preprocessed with one of the available phase
    computation function and must contain the following columns:

        - normalised channels (channel1 + "_NORM", etc)
        - cell cycle percentage
        - FRAME

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    channel1 : str
        First channel
    channel2 : str
        Second channel

    Raises
    ------
    ValueError
        If the dataframe does not contain the FRAME, CELL_CYCLE_PERC and normalised
        columns.
    """
    # check if the FRAME column is present
    if "FRAME" not in df.columns:
        raise ValueError("Column FRAME not found")

    # check if all new columns are present
    if NewColumns.cell_cycle() not in df.columns:
        raise ValueError(f"Column {NewColumns.cell_cycle()} not found")

    # get frame, normalised channels, unique intensity and phase
    t = df["FRAME"].to_numpy()
    channel1_norm = df[get_norm_channel_name(channel1)]
    channel2_norm = df[get_norm_channel_name(channel2)]
    unique_intensity = df[NewColumns.cell_cycle()]

    # plot
    plt.plot(t, channel1_norm, label=channel1)
    plt.plot(t, channel2_norm, label=channel2)
    plt.plot(t, unique_intensity, label="unique intensity")


def plot_dtw_query_vs_reference(
    reference_df: pd.DataFrame,
    df: pd.DataFrame,
    channels: List[str],
    ref_percentage_column: str = "percentage",
    est_percentage_column: str = "CELL_CYCLE_PERC_DTW",
    ground_truth: Optional[pd.DataFrame] = None,
    colors: Optional[List[str]] = None,
    **plot_kwargs: bool,
) -> None:
    """Plot query and alignment to reference curve.

    Parameters
    ----------
    reference_df: pd.DataFrame
        DataFrame with reference curve data
    df: pd.DataFrame
        DataFrame used for query
    channels: List[str]
        Name of the channels
    ref_percentage_column: str
        Name of column with percentages of reference curve
    est_percentage_column: str
        Name of column with estimated percentages
    ground_truth: pd.DataFrame
        DataFrame with ground truth data, needs to be named as reference_df
    colors: List[str]
        Colors for plot
    plot_kwargs: dict
        Kwargs to be passed to matplotlib
    """
    for channel in channels:
        if channel not in reference_df.columns:
            raise ValueError(f"Channel {channel} not in reference DataFrame")
        if channel not in df.columns:
            raise ValueError(f"Channel {channel} not in query DataFrame")
    if est_percentage_column not in df.columns:
        raise ValueError(
            "Percentage column not found in query DataFrame"
            f", available options {df.columns}"
        )
    if colors is None:
        colors = ["cyan", "magenta"]
    if ref_percentage_column not in reference_df.columns:
        raise ValueError(
            "Percentage column not found in reference DataFrame"
            f", available options {reference_df.columns}"
        )
    fig, ax = plt.subplots(1, len(channels))
    for idx, channel in enumerate(channels):
        ax[idx].plot(
            df[est_percentage_column], df[channel], label="Query", **plot_kwargs
        )
        ax[idx].plot(
            reference_df[ref_percentage_column],
            reference_df[channel],
            color=colors[idx],
            **plot_kwargs,
        )
        f_cyan = interpolate.interp1d(
            reference_df[ref_percentage_column], reference_df[channel]
        )
        ax[idx].plot(
            df[est_percentage_column],
            f_cyan(df[est_percentage_column]),
            lw=6,
            alpha=0.5,
            color="red",
            label="Match",
        )
        if ground_truth is not None:
            ax[idx].plot(
                ground_truth[ref_percentage_column],
                ground_truth[channel],
                label="Ground truth",
                lw=3,
            )

        ax[idx].set_ylabel(f"{channel.capitalize()} intensity / arb. u.")
        ax[idx].set_xlabel("Cell cycle percentage")
        if idx == 0:
            ax[idx].legend()
        plt.tight_layout()


def plot_query_vs_reference_in_time(
    reference_df: pd.DataFrame,
    df: pd.DataFrame,
    channels: List[str],
    ref_time_column: str = "time",
    query_time_column: str = "time",
    colors: Optional[List[str]] = None,
    channel_titles: Optional[List[str]] = None,
    fig_title: Optional[str] = None,
    **plot_kwargs: bool,
) -> None:
    """Plot query and alignment to reference curve.

    Parameters
    ----------
    reference_df: pd.DataFrame
        DataFrame with reference curve data
    df: pd.DataFrame
        DataFrame used for query
    channels: List[str]
        Name of the channels
    ref_time_column: str
        Name of column with times of reference curve
    query_time_column: str
        Name of column with times in query
    colors: List[str]
        Colors for plot
    plot_kwargs: dict
        Kwargs to be passed to matplotlib
    channel_titles: Optional[List]
        titles for each channel
    fig_title: Optional[str]
        Figure title
    """
    for channel in channels:
        if channel not in reference_df.columns:
            raise ValueError(f"Channel {channel} not in reference DataFrame")
        if channel not in df.columns:
            raise ValueError(f"Channel {channel} not in query DataFrame")
    if query_time_column not in df.columns:
        raise ValueError(
            f"Time column not found in query DataFrame, available options {df.columns}"
        )
    if channel_titles is not None:
        if len(channels) != len(channel_titles):
            raise ValueError("Provide a channel name for each channel")
    if ref_time_column not in reference_df.columns:
        raise ValueError(
            "Time column not found in reference DataFrame"
            f", available options {reference_df.columns}"
        )
    if colors is None:
        colors = ["cyan", "magenta"]
    fig, ax = plt.subplots(1, len(channels))
    if fig_title is not None:
        fig.suptitle(fig_title)
    for idx, channel in enumerate(channels):
        ax[idx].plot(
            df[query_time_column],
            df[channel],
            label="Query",
            color="blue",
            **plot_kwargs,
        )
        ax[idx].plot(
            reference_df[ref_time_column],
            reference_df[channel],
            color=colors[idx],
            **plot_kwargs,
        )
        ax[idx].set_yticks([])
        ax[idx].set_xlabel("Time / h")
        if idx == 0:
            ax[idx].set_ylabel("Intensity / arb. u.")
            ax[idx].legend()
        if channel_titles is not None:
            ax[idx].set_title(channel_titles[idx])
        plt.tight_layout()


def get_phase_color(phase: str) -> tuple:
    """Get color for a certain phase."""
    if phase == "G1":
        return (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0)
    elif phase == "S/G2/M":
        return (0.75, 0.0, 0.75, 1.0)
    else:
        return (0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 1.0)


def get_percentage_color(percentage: float) -> tuple:
    """Get color corresponding to percentage."""
    cmap_name = "cool"
    cmap = colormaps.get(cmap_name)
    if np.isnan(percentage):
        print("WARNING: NaN value detected, plot will be transparent")
        rgba_value = (0, 0, 0, 0)
    else:
        rgba_value = cmap(percentage / 100.0)
    return (rgba_value[0], rgba_value[1], rgba_value[2], 1.0)


def plot_cell_trajectory(
    track_df: pd.DataFrame,
    track_id_name: str,
    min_track_length: int = 30,
    centroid0_name: str = "centroid-0",
    centroid1_name: str = "centroid-1",
    phase_column: Optional[str] = None,
    percentage_column: Optional[str] = None,
    coloring_mode: str = "phase",
    line_cycle: Optional[list] = None,
    **kwargs: int,
) -> None:
    """Plot cell migration trajectories with phase or percentage-based coloring.

    Parameters
    ----------
    track_df : pandas.DataFrame
        DataFrame containing cell tracking data.
    track_id_name : str
        Column name containing unique track identifiers.
    min_track_length : int, optional
        Minimum number of timepoints required to include a track, default is 30.
    centroid0_name : str, optional
        Column name for x-coordinate of cell centroid, default is "centroid-0".
    centroid1_name : str, optional
        Column name for y-coordinate of cell centroid, default is "centroid-1".
    phase_column : str, optional
        Column name containing cell cycle phase information, default is None.
    percentage_column : str, optional
        Column name containing percentage values for coloring, default is None.
    coloring_mode : str, optional
        Color tracks by cell cycle phase (`phase`) or by percentage (`percentage`)
    line_cycle: list
        Cycle through the list, can help with visualization
    kwargs: dict, optional
        Kwargs are directly passed to the LineCollection, use it to adjust
        the linestyle for example

    Notes
    -----
    Phase or percentage columns need to be provided for the respective coloring.
    If not, an error will be raised.

    """
    # inital checks
    possible_coloring = ["phase", "percentage"]
    if coloring_mode not in possible_coloring:
        raise ValueError(f"coloring_mode needs to be one {possible_coloring}")

    if phase_column is None and coloring_mode == "phase":
        raise ValueError("No phase column value provided but phase coloring required.")
    if percentage_column is None and coloring_mode == "percentage":
        raise ValueError(
            "No percentage column value provided but percentage coloring required."
        )

    if "ls" in kwargs or "linestyles" in kwargs:
        raise ValueError("Set the linestyles via line_cycle argument.")
    # default: all curves solid
    if line_cycle is None:
        line_cycle = ["solid"]
    linecycler = cycle(line_cycle)
    # data structures
    line_collections = []

    # populate data structures
    indices = track_df[track_id_name].unique()
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for index in indices:
        if len(track_df.loc[track_df[track_id_name] == index]) < min_track_length:
            continue
        centroids = track_df.loc[
            track_df[track_id_name] == index, [centroid0_name, centroid1_name]
        ].to_numpy()
        # set start location to (0, 0)
        centroids[:, 0] -= centroids[0, 0]
        centroids[:, 1] -= centroids[0, 1]
        xmin = min(xmin, centroids[:, 0].min())
        xmax = max(xmax, centroids[:, 0].max())
        ymin = min(ymin, centroids[:, 1].min())
        ymax = max(ymax, centroids[:, 1].max())
        lines = np.c_[
            centroids[:-1, 0], centroids[:-1, 1], centroids[1:, 0], centroids[1:, 1]
        ]
        if phase_column is not None:
            phase_colors = (
                track_df.loc[track_df[track_id_name] == index, phase_column]
                .map(get_phase_color)
                .to_list()
            )
        if percentage_column is not None:
            phase_colors = (
                track_df.loc[track_df[track_id_name] == index, percentage_column]
                .map(get_percentage_color)
                .to_list()
            )

        line_collections.append(
            LineCollection(
                lines.reshape(-1, 2, 2),
                colors=phase_colors,
                ls=next(linecycler),
                **kwargs,
            )
        )
    fig, ax = plt.subplots()
    for line_collection in line_collections:
        ax.add_collection(line_collection)
    ax.margins(0.05)
    plt.xlabel(r"X in $\mu$m")
    plt.ylabel(r"Y in $\mu$m")
