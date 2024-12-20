from enum import Enum
from typing import List

import dtaidistance.preprocessing
import numpy as np
import pandas as pd
from dtaidistance.subsequence.dtw import subsequence_alignment
from scipy import interpolate, stats

from .sensor import FUCCISensor
from .utils import check_channels, check_thresholds, get_norm_channel_name


class NewColumns(str, Enum):
    """Columns generated by the analysis.


    Attributes
    ----------
    CELL_CYCLE_PERC : str
        Unique cell cycle percentage value
    PHASE : str
        Phase of the cell cycle
    """

    CELL_CYCLE_PERC_DTW = "CELL_CYCLE_PERC_DTW"
    CELL_CYCLE_PERC = "CELL_CYCLE_PERC"
    PHASE = "PHASE"
    DISCRETE_PHASE_MAX = "DISCRETE_PHASE_MAX"
    DISCRETE_PHASE_BG = "DISCRETE_PHASE_BG"
    DISCRETE_PHASE_DIFF = "DISCRETE_PHASE_DIFF"

    @staticmethod
    def cell_cycle() -> str:
        """Return the name of the unique intensity column."""
        return NewColumns.CELL_CYCLE_PERC.value

    @staticmethod
    def phase() -> str:
        """Return the name of the phase column."""
        return NewColumns.PHASE.value

    @staticmethod
    def cell_cycle_dtw() -> str:
        """Return the name of the cell cycle percentage column."""
        return NewColumns.CELL_CYCLE_PERC_DTW.value

    @staticmethod
    def discrete_phase_max() -> str:
        """Return the name of the discrete phase column."""
        return NewColumns.DISCRETE_PHASE_MAX.value

    @staticmethod
    def discrete_phase_bg() -> str:
        """Return the name of the discrete phase column."""
        return NewColumns.DISCRETE_PHASE_BG.value

    @staticmethod
    def discrete_phase_diff() -> str:
        """Return the name of the discrete phase column."""
        return NewColumns.DISCRETE_PHASE_DIFF.value


def generate_cycle_phases(
    df: pd.DataFrame, channels: List[str], sensor: FUCCISensor, thresholds: List[float]
) -> None:
    """Add a column in place to the dataframe with the phase of the cell cycle, where
    the phase is determined using a threshold on the cell cycle percentage.

    TODO update

    The thresholds must be between 0 and 1.
    The phase borders must be between 0 and 1.
    Each phase border is the expected percentage for the corresponding phase,
    the last phase will receive the difference to 1 (i.e., 100% of the cell cycle).

    Example:
        phases = ["G1", "S/G1", "SG2M"]
        phase_borders = [0.2, 0.2]
        thresholds = [0.1, 0.1]

    Here, SG2M spans the last 60% of the cell cycle.
    The thresholds mean that all intensities greater than 0.1 times the
    maximum intensity are considered ON.

    The phase borders need to be determined experimentally.
    Possible methods are FACS or analysis of the FUCCI intensities
    of multiple cell cycles recorded by live-cell fluorescence microscopy.

    The thresholds need to be chosen based on the expected noise of the background and
    uncertainty in intensity computation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns holding normalized intensities
    sensor: FUCCISensor
        FUCCI sensor with phase specifics
    channels: List[str]
        Names of channels
    thresholds: List[float]
        Thresholds to separate phases


    Raises
    ------
    ValueError
        If the number of thresholds is not 2
    ValueError
        If the phases are not unique
    ValueError
        If the thresholds are not between 0 and 1, one excluded
    """
    # sanity check: check that the normalized channels are present
    norm_channel_names = []
    for channel in channels:
        norm_channel_name = get_norm_channel_name(channel)
        if norm_channel_name not in df.columns:
            raise ValueError(
                f"Column {get_norm_channel_name(channel)} not found, call "
                f"normalize_channel({channel}) on the dataframe."
            )
        norm_channel_names.append(norm_channel_name)

    # check that all channels are present
    check_channels(sensor.fluorophores, channels)

    # compute phases
    estimate_cell_phase_from_max_intensity(
        df,
        norm_channel_names,
        sensor,
        background=[0] * sensor.fluorophores,
        thresholds=thresholds,
    )  # TODO check if background is correct

    # name of phase_column
    phase_column = NewColumns.discrete_phase_max()
    # compute percentages
    estimate_cell_cycle_percentage(df, norm_channel_names, sensor, phase_column)


def estimate_cell_cycle_percentage(
    df: pd.DataFrame, channels: List[str], sensor: FUCCISensor, phase_column: str
) -> None:
    """Estimate cell cycle percentage from intensity pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns holding normalized intensities
    sensor: FUCCISensor
        FUCCI sensor with phase specifics
    channels: List[str]
        Names of channels
    phase_column: str
        Name of phase column
    """
    percentages = []
    # iterate through data frame
    for _, row in df.iterrows():
        intensities = [row[channel] for channel in channels]
        phase = row[phase_column]
        percentage = sensor.get_estimated_cycle_percentage(phase, intensities)
        percentages.append(percentage)

    # TODO add inplace to dataframe
    # df[NewColumns.cell_cycle()] = pd.Series(percentages, dtype=float)
    df[NewColumns.cell_cycle()] = percentages


def estimate_cell_phase_from_max_intensity(
    df: pd.DataFrame,
    channels: List[str],
    sensor: FUCCISensor,
    background: List[float],
    thresholds: List[float],
) -> None:
    """Add a column in place to the dataframe with the estimated phase of the cell
    cycle, where the phase is determined by thresholding the channel intensities.

    The provided thresholds are used to decide if a channel is switched on (ON).
    For that, the background is subtracted from the mean intensity.
    The obtained values are normalized w.r.t. the maximum mean intensity in the
    respective channel available in the DataFrame.
    Hence, the threshold values should be between 0 and 1.
    This method will not work reliably if not enough cells from different phases
    are contained in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with a CELL_CYCLE_PERC column
    channels: List[str]
        Names of channels
    sensor: FUCCISensor
        FUCCI sensor with specific phase analysis information
    background: List[float]
        Single value per channel representing background
    thresholds: List[float]
        Thresholds to separate phases

    Raises
    ------
    ValueError
        If the dataframe does not contain the normalized channels.
    """
    # sanity check: check that channels are present
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(
                f"Column {channel} not found, provide correct input parameters."
            )

    if len(channels) != len(background):
        raise ValueError("Provide one background value per channel.")

    check_channels(sensor.fluorophores, channels)
    check_thresholds(sensor.fluorophores, thresholds)

    phase_markers_list: List["pd.Series[bool]"] = []
    for channel, bg_value, threshold in zip(channels, background, thresholds):
        # get intensities and subtract background
        intensity = df[channel] - bg_value
        # threshold channels to decide if ON / OFF (data is in list per spot)
        phase_markers_list.append(intensity > threshold * intensity.max())
    phase_markers_list_tilted = np.array(phase_markers_list).T

    # store phases
    phase_names = []
    for phase_markers in phase_markers_list_tilted:
        phase_names.append(sensor.get_phase(phase_markers))
    # TODO check pd.Series issue
    df[NewColumns.discrete_phase_max()] = phase_names


def estimate_cell_phase_from_background(
    df: pd.DataFrame,
    channels: List[str],
    sensor: FUCCISensor,
    background: List[float],
    thresholds: List[float],
) -> None:
    """Add a column in place to the dataframe with the estimated phase of the cell
    cycle, where the phase is determined by comparing the channel intensities to
    the respective background intensities.

    The provided factors are used to decide if a channel is switched on (ON).
    If the intensity exceeds the background level times the factor, the channel
    is ON. Hence, the factors should be greater than 0.


    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with a CELL_CYCLE_PERC column
    channels: List[str]
        Names of channels
    sensor: FUCCISensor
        FUCCI sensor with specific phase analysis information
    background: List[float]
        Single value per channel representing background
    thresholds: List[float]
        Thresholds to separate phases

    Raises
    ------
    ValueError
        If the dataframe does not contain the normalized channels.
    """
    # sanity check: check that channels are present
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(
                f"Column {channel} not found, provide correct input parameters."
            )

    if len(channels) != len(background):
        raise ValueError("Provide one background value per channel.")

    check_channels(sensor.fluorophores, channels)
    # TODO loosen check on boundary values (can be outside 0 to 1)
    # check_thresholds(sensor.fluorophores, thresholds)

    phase_markers_list: List["pd.Series[bool]"] = []
    for channel, bg_value, threshold in zip(channels, background, thresholds):
        intensity = df[channel]
        # threshold channels to decide if ON / OFF (data is in list per spot)
        phase_markers_list.append(intensity > threshold * bg_value)
    phase_markers_list_tilted = np.array(phase_markers_list).T

    # store phases
    phase_names = []
    for phase_markers in phase_markers_list_tilted:
        phase_names.append(sensor.get_phase(phase_markers))
    df[NewColumns.discrete_phase_bg()] = pd.Series(phase_names, dtype=str)  # add as str


# flake8: noqa: C901
def estimate_percentage_by_subsequence_alignment(
    df: pd.DataFrame,
    dt: float,
    channels: List[str],
    reference_data: pd.DataFrame,
    smooth: float = 0.1,
    penalty: float = 0.05,
    track_id_name: str = "TRACK_ID",
    minimum_track_length: int = 10,
) -> None:
    """Use subsequence alignment to estimate percentage.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with tracks
    dt: float
        Timestep between frames in hours
    channels: List[str]
        List of channels to be matched with reference data
    reference_data: pd.DataFrame
        Containing reference intensities over time
    smooth: float
        Smoothing factor, see dtaidistance documentation
    penalty: float
        Penalty for DTW algorithm, enforces diagonal warping path
    track_id_name: str
        Name of column with track IDs
    minimum_track_length: int
        Only estimate phase for tracks longer than this
    """
    if "time" not in reference_data:
        raise ValueError("Need to provide time column in reference_data.")
    if "percentage" not in reference_data:
        raise ValueError("Need to provide percentage column in reference_data.")

    if not set(channels).issubset(reference_data.columns):
        raise ValueError("Provide channel names in reference_data.")

    # interpolate reference curve
    time_scale = reference_data["time"].to_numpy()
    interpolation_functions = {}
    for channel in channels:
        interpolation_functions[channel] = interpolate.interp1d(
            time_scale, reference_data[channel].to_numpy()
        )
    f_percentage = interpolate.interp1d(
        time_scale, reference_data["percentage"].to_numpy()
    )

    num_time = int(time_scale[-1] / dt)
    new_time_scale = np.linspace(0, dt * num_time, num=num_time + 1)
    assert np.isclose(dt, new_time_scale[1] - new_time_scale[0])

    # reference curve in time scale of provided track
    percentage_ref = f_percentage(new_time_scale)

    series_diff = []
    for channel in channels:
        series = interpolation_functions[channel](new_time_scale)
        series = stats.zscore(series)
        series_diff.append(
            dtaidistance.preprocessing.differencing(series, smooth=smooth)
        )
    series = np.array(series_diff)
    series = np.swapaxes(series, 0, 1)

    df[NewColumns.cell_cycle_dtw()] = np.nan

    track_ids = df[track_id_name].unique()
    for track_id in track_ids:
        track_df = df.loc[df[track_id_name] == track_id]
        # the algorithm does not work for short tracks
        if len(track_df) < minimum_track_length:
            # insert NaN
            new_percentage = np.full(len(track_df), np.nan)
            df.loc[
                df[track_id_name] == track_id, NewColumns.cell_cycle_dtw()
            ] = new_percentage[:]
            continue

        # find percentages if track is long enough
        queries = track_df[channels].to_numpy()

        queries_diff = []
        for idx in range(len(channels)):
            queries[:, idx] = stats.zscore(queries[:, idx])
            queries_diff.append(
                dtaidistance.preprocessing.differencing(queries[:, idx], smooth=smooth)
            )

        query = np.array(queries_diff)
        query = np.swapaxes(query, 0, 1)

        sa = subsequence_alignment(query, series, penalty=penalty)
        best_match = sa.best_match()
        new_percentage = np.zeros(query.shape[0] + 1)
        for p in best_match.path:
            new_percentage[p[0]] = percentage_ref[p[1]]
        if p[1] + 1 < len(percentage_ref):
            last_percentage = p[1] + 1
        else:
            last_percentage = p[1]
        new_percentage[-1] = percentage_ref[last_percentage]
        df.loc[
            df[track_id_name] == track_id, NewColumns.cell_cycle_dtw()
        ] = new_percentage[:]
