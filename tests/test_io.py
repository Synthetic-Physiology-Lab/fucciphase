import pandas as pd
from fucciphase.io import read_trackmate_csv


def test_read_csv(trackmate_csv, trackmate_df: pd.DataFrame):
    """Read a dummy trackmate csv file, checking that it does
    not contains the first three lines (headers and units)."""
    df = read_trackmate_csv(trackmate_csv)

    assert df.equals(trackmate_df)
