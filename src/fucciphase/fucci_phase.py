from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .io import read_trackmate_xml
from .phase import generate_cycle_phases
from .utils import normalize_channels


def process_dataframe(
    df: pd.DataFrame,
    g1_channel: str,
    s_g2_channel: str,
    phases: List[str],
    phase_borders: List[float],
    thresholds: List[float],
    use_moving_average: bool = True,
    window_size: int = 7,
    manual_min: Optional[List[float]] = None,
    manual_max: Optional[List[float]] = None,
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
    g1_channel : str
        Smaller wavelength FUCCI channel
    s_g2_channel : str
        Longer wavelength FUCCI channel
    use_moving_average : bool, optional
        Use moving average before normalization, by default True
    window_size : int, optional
        Window size of the moving average, by default 5
    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None
    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None
    phases : Optional[List[str]], optional
        List of the different phases, by default None
    phase_borders : Optional[List[float]], optional
        List of the percentages marking the end of each phase, should
        one element less than the number of phases
    thresholds : Optional[List[float]], optional
        List of thresholds to be used for each channel to decide if it is ON or OFF
    """
    # normalize the channels
    normalize_channels(
        df,
        [g1_channel, s_g2_channel],
        use_moving_average=use_moving_average,
        moving_average_window=window_size,
        manual_min=manual_min,
        manual_max=manual_max,
    )

    # compute the phases
    generate_cycle_phases(
        df,
        g1_channel,
        s_g2_channel,
        phases=phases,
        phase_borders=phase_borders,
        thresholds=thresholds,
    )


def process_trackmate(
    xml_path: Union[str, Path],
    g1_channel: str,
    s_g2_channel: str,
    phases: List[str],
    phase_borders: List[float],
    thresholds: List[float],
    use_moving_average: bool = True,
    window_size: int = 5,
    manual_min: Optional[List[float]] = None,
    manual_max: Optional[List[float]] = None,
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
        - compute the cell cycle percentage using a trigonometric approach
        - if `phases` and `thresholds` are not None, compute the phases of the cell
        - save an updated XML copy with the new features

    # TODO: add details about the normalization, the trigonometric approach and phases

    Parameters
    ----------
    xml_path : Union[str, Path]
        Path to the XML file
    g1_channel : str
        Smaller wavelength FUCCI channel
    s_g2_channel : str
        Longer wavelength FUCCI channel
    use_moving_average : bool, optional
        Use moving average before normalization, by default True
    window_size : int, optional
        Window size of the moving average, by default 5
    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None
    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None
    phases : Optional[List[str]], optional
        List of the different phases, by default None
    phase_borders : Optional[List[float]], optional
        List of the percentages marking the end of each phase, should
        one element less than the number of phases
    thresholds : Optional[List[float]], optional
        List of thresholds to be used for each channel to decide if it is ON or OFF

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
        g1_channel=g1_channel,
        s_g2_channel=s_g2_channel,
        use_moving_average=use_moving_average,
        window_size=window_size,
        manual_min=manual_min,
        manual_max=manual_max,
        phases=phases,
        phase_borders=phase_borders,
        thresholds=thresholds,
    )

    # update the XML
    tmxml.update_features(df)

    # export the xml
    new_name = Path(xml_path).stem + "_processed.xml"
    new_path = Path(xml_path).parent / new_name
    tmxml.save_xml(new_path)

    return df
