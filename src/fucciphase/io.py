from pathlib import Path

import pandas as pd

from .utils import TrackMateXML


def read_trackmate_xml(xml_path: Path | str) -> tuple[pd.DataFrame, TrackMateXML]:
    """Read a TrackMate-exported XML file and return data and XML wrapper.

    Parameters
    ----------
    xml_path : Union[Path, str]
        Path to the XML file.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the spot and track data, sorted by FRAME.
    trackmate : TrackMateXML
        TrackMateXML object wrapping the original XML and allowing
        feature updates / re-export.
    """
    # read in the xml file
    trackmate = TrackMateXML(xml_path)

    # convert the spots to a dataframe
    df = trackmate.to_pandas()
    # sort by frame number to have increasing time
    df.sort_values(by="FRAME")

    return df, trackmate


def read_trackmate_csv(csv_path: Path | str) -> pd.DataFrame:
    """Read a TrackMate-exported CSV file.

    The first three rows (excluding header) of the csv file are skipped as
    they contain duplicate titles of columns and units (Trackmate specific).

    The first three rows (excluding the header) of the CSV file are
    skipped as they contain duplicate column titles and units
    (TrackMate-specific).

    Parameters
    ----------
    csv_path : Union[Path, str]
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the CSV data with converted dtypes.

    Raises
    ------
    ValueError
        If the CSV file does not contain both MEAN_INTENSITY_CH1 and
        MEAN_INTENSITY_CH2 columns.
    """
    df = pd.read_csv(csv_path, encoding="unicode_escape", skiprows=[1, 2, 3])

    # sanity check: trackmate must have at least two channels
    if (
        "MEAN_INTENSITY_CH1" not in df.columns
        and "MEAN_INTENSITY_CH2" not in df.columns
    ):
        raise ValueError("Trackmate must have at least two channels.")

    # return dataframe with converted types (object -> string)
    return df.convert_dtypes()
