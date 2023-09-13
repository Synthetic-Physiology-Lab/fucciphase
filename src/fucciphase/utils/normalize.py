from typing import List, Optional, Union

import numpy as np
import pandas as pd


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


# TODO: is there a simpler way? The convolution is probably an advantage for large v
def moving_average(vector: Union[pd.Series, np.ndarray], window: int = 7) -> np.ndarray:
    """Compute the moving average of a vector using a fixed window.

    The returned vector has the same size as the input vector. The moving average
    function breaks the problem in two parts:
        - Edges: computed using a an increasing window size (starting from w//2)
        - Middle: computed using a fixed window size (w)

    The window size must be odd.

    Parameters
    ----------
    vector : Union[pd.Series, np.ndarray]
        Vector to average.
    window : int
        Size of the window, must be odd.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        Averaged vector.
    """
    if window < 1:
        raise ValueError(f"Window size must be > 0, got {window}")

    if window % 2 == 0:
        raise ValueError(f"Window size must be odd, got {window}")

    # if the window is larger than the vector, return the vector
    if window >= len(vector):
        return vector

    # compute the cumulative sum
    cumsum = np.cumsum(vector)

    # compute left-hand edge using the cumulative sum
    # index i starts at 0, and ends when the window is one index away from fitting on
    # the left-hand side
    # cumsum is the cumulative sum of the whole vector
    # the window size increases with the indices
    pre_vector = np.array([cumsum[i] / (i + 1) for i in range(window // 2, window - 1)])

    # compute right-hand edge using the cumulative sum
    # index i starts when the window does not fit anymore in the right-hand side
    # therefore i starts at len(vector) - window//2, and ends at the last index
    # cumsum[-1] is the sum of the whole vector
    # cumsum[i - window//2 - 1] is the sum until the index before the window starts
    # the difference between the two is the sum of the window
    # we divide by the number of elements in the window, the window reduces in size
    # for each index
    post_vector = np.array(
        [
            (cumsum[-1] - cumsum[i - window // 2 - 1]) / (window // 2 + len(vector) - i)
            for i in range(len(vector) - window // 2, len(vector))
        ]
    )

    # compute valid convolution for the middle part
    middle_vector = np.array(vector, dtype=float)
    middle_vector = np.convolve(middle_vector, np.ones(window), mode="valid") / window

    return np.concatenate([pre_vector, middle_vector, post_vector], dtype=float)


def normalize_channels(
    df: pd.DataFrame,
    channels: Union[str, List[str]],
    use_moving_average: bool = False,
    moving_average_window: int = 7,
    manual_min: Optional[float] = None,
    manual_max: Optional[float] = None,
) -> List[str]:
    """Normalize channels, add in place the resulting columns to the
    dataframe, and return the new columns' name.

    A moving average can be applied to each individual track before normalization.

    Normalization is performed by subtracting the min and dividing by (max - min).
    Note that the resulting values are rounded to the 2nd decimal.

    The min and max values can be provided manually. They should be determined by
    imaging a large number of cells statically and computing the min and max values
    observed.

    TODO: add not about ergodicity, more details about static imaging

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
    manual_min : Optional[float]
        If provided, the minimum value to use for normalization.
    manual_max : Optional[float]
        If provided, the maximum value to use for normalization.

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

    # check that the dataframe contains the channek
    new_columns = []
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(f"Column {channel} not found")

        # apply moving average to each track ID
        unique_track_IDs = df["TRACK_ID"].unique()

        # compute the moving average for each track ID
        if use_moving_average:
            avg_channel = get_avg_channel_name(channel)
            for track_ID in unique_track_IDs:
                # get the track
                track: pd.DataFrame = df[df["TRACK_ID"] == track_ID]

                # sort the channel by frame
                track = track.sort_values(by="FRAME")

                # compute the moving average
                ma = moving_average(track[channel], window=moving_average_window)

                # update the dataframe by adding a new column
                df.loc[track.index, avg_channel] = ma
        else:
            avg_channel = channel

        # normalize channel
        max_ch = df[avg_channel].max()
        min_ch = df[avg_channel].min()
        norm_ch = np.round(
            (df[avg_channel] - min_ch) / (max_ch - min_ch),
            2,  # number of decimals
        )

        # add the new column
        new_column = get_norm_channel_name(channel)
        df[new_column] = norm_ch
        new_columns.append(new_column)

    return new_columns
