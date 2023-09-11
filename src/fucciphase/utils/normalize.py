from typing import List, Union

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


def normalize_channels(df: pd.DataFrame, channels: Union[str, List[str]]) -> List[str]:
    """Normalize channels, add in place the resulting columns to the
    dataframe, and return the new columns' name.

    Normalization is performed by subtracting the min and dividing by (max - min).
    Note that the resulting values are rounded to the 2nd decimal.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    channels : Union[str, List[str]]
        Name of the channels to normalize.

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

        # normalize channel
        max_ch = df[channel].max()
        min_ch = df[channel].min()
        norm_ch = np.round(
            (df[channel] - min_ch) / (max_ch - min_ch),
            2,  # number of decimals
        )

        # add the new column
        new_column = get_norm_channel_name(channel)
        df[new_column] = norm_ch
        new_columns.append(new_column)

    return new_columns
