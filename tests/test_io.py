import pandas as pd
from fucciphase.io import read_trackmate_csv, read_trackmate_xml


def test_read_csv(trackmate_csv, trackmate_df: pd.DataFrame):
    """Read a dummy trackmate csv file, checking that it does
    not contains the first three lines (headers and units)."""
    df = read_trackmate_csv(trackmate_csv)

    assert df.equals(trackmate_df)


def test_read_xml(trackmate_xml):
    """Read a simple Trackmate XML file and check that it returned
    the correct data."""
    df, tmxml = read_trackmate_xml(trackmate_xml)

    # check the dataframe
    assert len(df) == tmxml.nspots == 4
