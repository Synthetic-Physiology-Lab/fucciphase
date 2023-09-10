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
    """
    df = pd.read_csv(csv_path, encoding="unicode_escape", skiprows=[1, 2, 3])

    # return dataframe with converted types (object -> string)
    return df.convert_dtypes()
