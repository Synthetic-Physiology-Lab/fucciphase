import pandas as pd
from fucciphase.io import read_trackmate_csv


def test_read_csv(trackmate_csv, trackmate_df: pd.DataFrame):
    """Read a dummy trackmate csv file, checking that it does
    not contains the first three lines (headers and units)."""
    df = read_trackmate_csv(trackmate_csv)

    # remove the first three rows, re-index the dataframe and
    # convert the types
    trackmate_df.drop(index=[0, 1, 2], inplace=True)
    trackmate_df.reset_index(drop=True, inplace=True)
    trackmate_df = trackmate_df.convert_dtypes()

    assert df.equals(trackmate_df)
