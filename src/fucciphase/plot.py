from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from .phase import NewColumns
from .utils import get_norm_channel_name


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
