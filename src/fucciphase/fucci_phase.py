from pathlib import Path

import pandas as pd

from .io import read_trackmate_xml
from .phase import generate_cycle_phases
from .sensor import FUCCISensor
from .utils import normalize_channels, split_trackmate_tracks


def process_dataframe(
    df: pd.DataFrame,
    channels: list[str],
    sensor: FUCCISensor,
    thresholds: list[float],
    use_moving_average: bool = True,
    window_size: int = 7,
    manual_min: list[float] | None = None,
    manual_max: list[float] | None = None,
    generate_unique_tracks: bool = False,
    track_id_name: str = "TRACK_ID",
    label_id_name: str = "name",
    estimate_percentage: bool = True,
) -> None:
    """Apply the FUCCIphase analysis pipeline to an existing dataframe.

    This function assumes that tracking and fluorescence information are
    already available in a pandas DataFrame with the expected column
    structure. It performs the same core steps as ``process_trackmate``,
    but skips the TrackMate file I/O and starts directly from tabular data.

    Use this when your tracking pipeline already provides a dataframe, or
    when you have manually assembled the input table and still want to use
    FUCCIphase for cell-cycle analysis and visualization.

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
        Input dataframe containing tracking and intensity features.

    channels: List[str]
         Names of columns holding FUCCI fluorescence information.

    sensor : FUCCISensor
        FUCCI sensor with phase-specific parameters.

    thresholds: List[float]
        Thresholds used to separate cell-cycle phases.

    use_moving_average : bool, optional
        If True, apply a moving average before normalization. Default is True.

    window_size : int, optional
        Window size of the moving average. Default is 7.

    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None

    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None

    generate_unique_tracks: bool
        If True, assign unique track IDs to split tracks. This requires
        using the appropriate action in TrackMate. Default is False.

    track_id_name: str
        Name of the column containing track IDs. Default is ``"TRACK_ID"``.

    label_id_name: str
        Column name identifying the spot label / name (used for unique
        track ID generation). Default is ``"name"``.

    estimate_percentage: bool, optional
        If True, estimate cell-cycle percentage along each track. Default is True.

    Returns
    -------
    None
        The input dataframe is modified in-place. No value is returned.
    """
    # ensure that the number of provided channels matches the sensor definition
    if len(channels) != sensor.fluorophores:
        raise ValueError(f"Need to provide {sensor.fluorophores} channel names.")

    # optionally split TrackMate subtracks and re-label them as unique tracks
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

    # compute the phases (and, optionally, the cycle percentage)
    generate_cycle_phases(
        df,
        sensor=sensor,
        channels=channels,
        thresholds=thresholds,
        estimate_percentage=estimate_percentage,
    )


def process_trackmate(
    xml_path: str | Path,
    channels: list[str],
    sensor: FUCCISensor,
    thresholds: list[float],
    use_moving_average: bool = True,
    window_size: int = 7,
    manual_min: list[float] | None = None,
    manual_max: list[float] | None = None,
    generate_unique_tracks: bool = False,
    estimate_percentage: bool = True,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full FUCCIphase pipeline on a TrackMate export.

    This high-level helper takes tracking data exported from Fiji/TrackMate
    (typically XML or CSV), converts it into a pandas DataFrame with the
    expected fucciphase columns, applies basic quality checks and
    preprocessing, and estimates cell-cycle phase information that can be
    used for downstream analysis and plotting.

    The returned table is intended to be the main entry point for
    fucciphase workflows, and is compatible with the plotting and
    visualization functions provided in this package.

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
        Path to the TrackMate XML file.
    channels : List[str]
        Names of columns holding FUCCI fluorescence information.
    sensor : FUCCISensor
        FUCCI sensor with phase-specific parameters.
    thresholds : List[float]
        Thresholds used to separate cell-cycle phases.
    use_moving_average : bool, optional
        If True, apply a moving average before normalization. Default is True.
    window_size : int, optional
        Window size of the moving average. Default is 7.
    manual_min : Optional[List[float]], optional
        Manually determined minimum for each channel, by default None.
    manual_max : Optional[List[float]], optional
        Manually determined maximum for each channel, by default None.
    generate_unique_tracks : bool, optional
        If True, assign unique track IDs to split tracks. This requires
        using the appropriate action in TrackMate. Default is False.
    estimate_percentage : bool, optional
        If True, estimate cell-cycle percentage along each track. Default is True.
    output_dir : Optional[Union[str, Path]], optional
        Optional directory where the updated XML should be written. If None,
        the file is saved next to the input XML.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the cell-cycle percentage and the corresponding phases.

    """
    # read the XML and extract the dataframe and XML wrapper
    df, tmxml = read_trackmate_xml(xml_path)

    # process the dataframe in-place (and also get a reference to it)
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

    # update the XML with the new features
    tmxml.update_features(df)

    # export the updated XML next to the original file
    new_name = Path(xml_path).stem + "_processed.xml"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / new_name
    else:
        new_path = Path(xml_path).parent / new_name

    tmxml.save_xml(new_path)

    return df
