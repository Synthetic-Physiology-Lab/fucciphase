from pathlib import Path
from typing import Union

import pandas as pd


def read_trackmate_csv(csv_path: Union[Path, str]) -> pd.DataFrame:
    """Read a trackmate exported csv file.

    The first three rows (excluding header) of the csv file are skipped as
    they contain duplicate titles of columns and units (Trackmate specific).


    Parameters
    ----------
    csv_path : str
        Path to the csv file.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the csv data.

    Raises
    ------
    ValueError
        If the csv file does not contain at least two channels.
    """
    df = pd.read_csv(csv_path, encoding="unicode_escape", skiprows=[1, 2, 3])

    # sanity check: trackmate must have at least two channels
    # TODO should it have 4?
    if (
        "MEAN_INTENSITY_CH1" not in df.columns
        and "MEAN_INTENSITY_CH2" not in df.columns
    ):
        raise ValueError("Trackmate must have at least two channels.")

    # return dataframe with converted types (object -> string)
    return df.convert_dtypes()
