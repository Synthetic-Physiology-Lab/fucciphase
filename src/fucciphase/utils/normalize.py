from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal


def get_norm_channel_name(channel: str) -> str:
    """Return the name of the normalized channel.

    Parameters
    ----------
    channel : str
        Name of the channel to normalize.

    Returns
    -------
    str
        Name of the normalized channel.
    """
    return f"{channel}_NORM"


def get_avg_channel_name(channel: str) -> str:
    """Return the name of the moving-averaged channel.

    Parameters
    ----------
    channel : str
        Name of the channel to average using moving average.

    Returns
    -------
    str
        Name of the moving-averaged channel.
    """
    return f"{channel}_AVG"


def norm(vector: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Normalize a vector by subtracting the min and dividing by (max - min).

    Parameters
    ----------
    vector : Union[pd.Series, np.ndarray]
        Vector to normalize.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        Normalized vector.
    """
    max_ch = vector.max()
    min_ch = vector.min()
    norm_ch = np.round(
        (vector - min_ch) / (max_ch - min_ch),
        2,  # number of decimals
    )

    return norm_ch


# flake8: noqa: C901
def normalize_channels(
    df: pd.DataFrame,
    channels: Union[str, List[str]],
    use_moving_average: bool = True,
    moving_average_window: int = 7,
    manual_min: Optional[List[float]] = None,
    manual_max: Optional[List[float]] = None,
    track_id_name: str = "TRACK_ID",
) -> List[str]:
    """Normalize channels, add in place the resulting columns to the
    dataframe, and return the new columns' name.

    A moving average can be applied to each individual track before normalization.

    Normalization is performed by inferring the min at the position of the maximum
    of the other channel. Then, the min is subtracted and the result is divided
    by (max - min).
    These values are computed across all spots in each channel. Note that the resulting
    normalized values are rounded to the 2nd decimal.

    The min and max values can be provided manually. They should be determined by
    imaging a large number of cells statically and computing the min and max values
    observed.
    This option is meant for static imaging. It is assumed that there are enough cells
    in the image to have enough samples from each phase of the cell cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    channels : Union[str, List[str]]
        Name of the channels to normalize.
    use_moving_average : bool
        Whether to apply a moving average to each track before normalization.
    moving_average_window : int
        Size of the window used for the moving average, default 7.
    manual_min : Optional[List[float]]
        If provided, the minimum value to use for normalization.
    manual_max : Optional[List[float]]
        If provided, the maximum value to use for normalization.
    track_id_name: str
        Name of column with track IDs

    Returns
    -------
    List[str]
        Name of the new column(s).

    Raises
    ------
    ValueError
        If the dataframe does not contain the mandatory columns.
    """
    if not isinstance(channels, list):
        channels = [channels]

    if manual_min is not None:
        # check that it has the same number of entries as there are channels
        if len(manual_min) != len(channels):
            raise ValueError(
                f"Expected {len(channels)} values for manual_min, got {len(manual_min)}"
            )
    if manual_max is not None:
        # check that it has the same number of entries as there are channels
        if len(manual_max) != len(channels):
            raise ValueError(
                f"Expected {len(channels)} values for manual_max, got {len(manual_max)}"
            )

    # check that the dataframe contains the channel
    new_columns = []
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(f"Column {channel} not found")

        # compute the moving average for each track ID
        if use_moving_average:
            # apply moving average to each track ID
            unique_track_IDs = df[track_id_name].unique()

            avg_channel = get_avg_channel_name(channel)
            for track_ID in unique_track_IDs:
                index, ma = smooth_track(
                    df, track_ID, channel, track_id_name, moving_average_window
                )
                # update the dataframe by adding a new column
                df.loc[index, avg_channel] = ma

    # normalize channels
    for channel in channels:
        # moving average creates a new column with an own name
        if use_moving_average:
            avg_channel = get_avg_channel_name(channel)
        else:
            avg_channel = channel
        # normalize channel
        norm_ch = norm(df[avg_channel])

        # add the new column
        new_column = get_norm_channel_name(channel)
        df[new_column] = norm_ch
        new_columns.append(new_column)

    return new_columns


def smooth_track(
    df: pd.DataFrame,
    track_ID: int,
    channel: str,
    track_id_name: str,
    moving_average_window: int = 7,
) -> Tuple[pd.Index, np.ndarray]:
    """Smooth intensity in one channel for a single track.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    track_ID: int
        Index of track
    channel : st
        Name of the channel to smooth
    track_id_name: str
        Name of column with track IDs
    moving_average_window : int
        Size of the window used for the moving average, default 7.
    """
    # get the track
    track: pd.DataFrame = df[df[track_id_name] == track_ID]

    # compute the moving average
    ma = signal.savgol_filter(
        track[channel],
        window_length=moving_average_window,
        polyorder=3,
        mode="nearest",
    )
    return track.index, ma
