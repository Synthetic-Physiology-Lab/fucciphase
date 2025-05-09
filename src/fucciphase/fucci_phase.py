from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .io import read_trackmate_xml
from .phase import generate_cycle_phases
from .sensor import FUCCISensor
from .utils import normalize_channels, split_trackmate_tracks


def process_dataframe(
    df: pd.DataFrame,
    channels: List[str],
    sensor: FUCCISensor,
    thresholds: List[float],
    use_moving_average: bool = True,
    window_size: int = 7,
    manual_min: Optional[List[float]] = None,
    manual_max: Optional[List[float]] = None,
    generate_unique_tracks: bool = False,
    track_id_name: str = "TRACK_ID",
    label_id_name: str = "name",
    estimate_percentage: bool = True,
) -> None:
    """Process a pd.DataFrame by computing the cell cycle percentage from two FUCCI
    cycle reporter channels in place.

    The dataframe must contain ID and TRACK_ID features.

    This function applies the following steps:
        - if `use_moving_average` is True, apply a Savitzky-Golay filter to each track
          and each channel
        - if `manual_min` and `manual_max` are None, normalize the channels globally.
          Otherwise, use them to normalize each channel.
        - compute the cell cycle phases and their estimated percentage

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe
    channels: List[str]
        Names of channels holding FUCCI information
    sensor : FUCCISensor
        FUCCI sensor with phase specifics
    thresholds: List[float]
        Thresholds to separate phases
    use_moving_average : bool, optional
        Use moving average before normalization, by default True
    window_size : int, optional
        Window size of the moving average, by default 5
    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None
    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None
    estimate_percentage: bool
        Estimate cell cycle percentage
    label_id_name: str
        Give an indentifier for the spot name (needed for unique track ID generation)
    generate_unique_tracks: bool
        Assign unique track IDs to splitted tracks.
        Requires usage of action in TrackMate.
    track_id_name: str
        Name of column with track IDs
    """
    if len(channels) != sensor.fluorophores:
        raise ValueError(f"Need to provide {sensor.fluorophores} channel names.")

    if generate_unique_tracks:
        if "TRACK_ID" in df.columns:
            split_trackmate_tracks(df, label_id_name=label_id_name)
            # perform all operation on unique tracks
            track_id_name = "UNIQUE_TRACK_ID"
        else:
            print("Warning: unique tracks can only be prepared for TrackMate files.")
            print("The tracks have not been updated.")

    # normalize the channels
    normalize_channels(
        df,
        channels,
        use_moving_average=use_moving_average,
        moving_average_window=window_size,
        manual_min=manual_min,
        manual_max=manual_max,
        track_id_name=track_id_name,
    )

    # compute the phases
    generate_cycle_phases(
        df,
        sensor=sensor,
        channels=channels,
        thresholds=thresholds,
        estimate_percentage=estimate_percentage,
    )


def process_trackmate(
    xml_path: Union[str, Path],
    channels: List[str],
    sensor: FUCCISensor,
    thresholds: List[float],
    use_moving_average: bool = True,
    window_size: int = 7,
    manual_min: Optional[List[float]] = None,
    manual_max: Optional[List[float]] = None,
    generate_unique_tracks: bool = False,
    estimate_percentage: bool = True,
) -> pd.DataFrame:
    """Process a trackmate XML file, compute cell cycle percentage from two FUCCI cycle
    reporter channels, save an updated copy of the XML and return the results in a
    dataframe.

    This function applies the following steps:
        - load the XML file and generate a dataframe from the spots and tracks
        - if `use_moving_average` is True, apply a Savitzky-Golay filter to each track
          and each channel
        - if `manual_min` and `manual_max` are None, normalize the channels globally.
          Otherwise, use them to normalize each channel.
        - compute the cell cycle percentage
        - save an updated XML copy with the new features

    Parameters
    ----------
    xml_path : Union[str, Path]
        Path to the XML file
    channels: List[str]
        Names of channels holding FUCCI information
    generate_unique_tracks: bool
        Assign unique track IDs to splitted tracks.
        Requires usage of action in TrackMate.
    sensor : FUCCISensor
        FUCCI sensor with phase specifics
    thresholds: List[float]
        Thresholds to separate phases
    use_moving_average : bool, optional
        Use moving average before normalization, by default True
    window_size : int, optional
        Window size of the moving average, by default 5
    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None
    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None
    estimate_percentage: bool, optional
        Estimate cell cycle percentage

    Returns
    -------
    pd.DataFrame
        Dataframe with the cell cycle percentage and the corresponding phases
    """
    # read the XML
    df, tmxml = read_trackmate_xml(xml_path)

    # process the dataframe
    process_dataframe(
        df,
        channels,
        sensor,
        thresholds,
        use_moving_average=use_moving_average,
        window_size=window_size,
        manual_min=manual_min,
        manual_max=manual_max,
        generate_unique_tracks=generate_unique_tracks,
        estimate_percentage=estimate_percentage,
    )

    # update the XML
    tmxml.update_features(df)

    # export the xml
    new_name = Path(xml_path).stem + "_processed.xml"
    new_path = Path(xml_path).parent / new_name
    tmxml.save_xml(new_path)

    return df
