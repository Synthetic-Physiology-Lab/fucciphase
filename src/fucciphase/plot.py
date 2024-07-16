from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

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
    track_name: str = "TRACK_ID",
    ylim: Optional[tuple] = None,
    yticks: Optional[list] = None,
) -> Figure:
    """Plot features of individual tracks in one plot."""
    if feature_name not in df:
        raise ValueError(f"(Feature {feature_name} not in provided DataFrame.")
    if time_column not in df:
        raise ValueError(f"(Time {time_column} not in provided DataFrame.")
    tracks = df[track_name].unique()
    tracks = tracks[tracks >= 0]

    fig = plt.figure()
    # Plot each graph, and manually set the y tick values
    for track_idx in tracks:
        time = df.loc[df[track_name] == track_idx, time_column].to_numpy()
        feature = df.loc[df[track_name] == track_idx, feature_name].to_numpy()
        plt.plot(time, feature)
        if ylim is not None:
            plt.ylim(ylim)
        if yticks is not None:
            plt.yticks(yticks)
    # TODO interplolate and plot average
    """
    plt.plot(interpolated_time, np.nanmean(interpolated_feature, lw=5, color="black")
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
    """
    return fig


# flake8: noqa: C901
def plot_feature_stacked(
    df: pd.DataFrame,
    time_column: str,
    feature_name: str,
    interpolate_time: bool = False,
    track_name: str = "TRACK_ID",
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
    tracks = df[track_name].unique()
    tracks = tracks[tracks >= 0]
    if selected_tracks is None:
        selected_tracks = tracks
    else:
        if not set(selected_tracks).issubset(tracks):
            raise ValueError(
                "Selected tracks contain tracks " "that are not in track list."
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
        time = df.loc[df[track_name] == track_idx, time_column].to_numpy()
        feature = df.loc[df[track_name] == track_idx, feature_name].to_numpy()
        colors = df.loc[df[track_name] == track_idx, "COLOR"].to_numpy()
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
            time = df.loc[df[track_name] == track_idx, time_column].to_numpy()
            feature = df.loc[df[track_name] == track_idx, feature_name].to_numpy()
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
    color1: Optional[str] = None,
    color2: Optional[str] = None,
    time_column: str = "FRAME",
    time_label: str = "Frame #",
    **plot_kwargs: bool,
) -> None:
    """TODO description."""
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
    color1: Optional[str] = None,
    color2: Optional[str] = None,
    time_column: str = "FRAME",
    time_label: str = "Frame #",
    **plot_kwargs: bool,
) -> None:
    """TODO description."""
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
