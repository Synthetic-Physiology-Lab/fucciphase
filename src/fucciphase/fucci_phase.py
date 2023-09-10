from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd

from .io import GeneratedColumns, MandatoryColumns


class Phase(str, Enum):
    """Phase of the cell cycle."""

    EARLY_G1 = "EG1"
    G1 = "G1"
    S = "T"  # TODO: why not S?
    G2_M = "G2/M"


class FucciPhases:
    """FUCCI phases normalized intensity.

    TODO: add explanations.
    """

    start_early_G1 = 0
    end_early_G1 = 4
    start_G1 = 5
    end_G1 = 95
    start_S = 96
    end_S = 113
    start_G2_M = 114
    end_G2_M = 255

    @staticmethod
    def get_phase(ch3_value: int, ch4_value: int) -> Tuple[Phase, int, int]:
        """Return the phase of the cell cycle, and its intensity range.

        Parameters
        ----------
        ch3_value : int
            Normalized intensity of channel 3.
        ch4_value : int
            Normalized intensity of channel 4.

        Returns
        -------
        Tuple[Phase, int, int]
            The phase of the cell cycle, and its intensity range (start and end).
        """
        if ch3_value <= 0.1 and ch4_value <= 0.1:
            return Phase.EARLY_G1, FucciPhases.start_early_G1, FucciPhases.end_early_G1
        elif ch3_value <= 0.1 and ch4_value > 0.1:
            return Phase.G1, FucciPhases.start_G1, FucciPhases.end_G1
        elif ch3_value > 0.1 and ch4_value > 0.1:
            return Phase.S, FucciPhases.start_S, FucciPhases.end_S
        else:
            return Phase.G2_M, FucciPhases.start_G2_M, FucciPhases.end_G2_M


def normalize_channels(df: pd.DataFrame) -> None:
    """Normalize the MEAN_INTENSITY_CH3 and MEAN_INTENSITY_CH4
    channels, and add the resulting columns to the dataframe in
    place.

    Normalization is performed by subtracting the min and
    dividing by (max - min).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Raises
    ------
    ValueError
        If the dataframe does not contain the mandatory columns.
    """
    # check that the dataframe contains the mandatory columns
    if MandatoryColumns.MEAN_INTENSITY_CH3 not in df.columns:
        raise ValueError(f"Column {MandatoryColumns.MEAN_INTENSITY_CH3} not found")
    if MandatoryColumns.MEAN_INTENSITY_CH4 not in df.columns:
        raise ValueError(f"Column {MandatoryColumns.MEAN_INTENSITY_CH4} not found")

    # normalize channel 3
    max_ch3 = df[MandatoryColumns.MEAN_INTENSITY_CH3].max()
    min_ch3 = df[MandatoryColumns.MEAN_INTENSITY_CH3].min()
    norm_ch3 = np.round(
        (df[MandatoryColumns.MEAN_INTENSITY_CH3] - min_ch3) / (max_ch3 - min_ch3),
        2,  # number of decimals
    )
    df[GeneratedColumns.MEAN_INTENSITY_CH3_NORM] = norm_ch3

    # normalize channel 4
    max_ch4 = df[MandatoryColumns.MEAN_INTENSITY_CH4].max()
    min_ch4 = df[MandatoryColumns.MEAN_INTENSITY_CH4].min()
    norm_ch4 = np.round(
        (df[MandatoryColumns.MEAN_INTENSITY_CH4] - min_ch4) / (max_ch4 - min_ch4),
        2,  # number of decimals
    )
    df[GeneratedColumns.MEAN_INTENSITY_CH4_NORM] = norm_ch4


def compute_phase_color(df: pd.DataFrame) -> None:
    """Compute the phase color for each spot, and update the dataframe
    with a unified mean intensity value, phase, and RGB color columns.



    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Raises
    ------
    ValueError
        If the dataframe does not contain the normalized channels.
    """
    # sanity check
    if GeneratedColumns.MEAN_INTENSITY_CH3_NORM not in df.columns:
        raise ValueError(
            f"Column {GeneratedColumns.MEAN_INTENSITY_CH3_NORM} not found, call"
            f"normalize_channels() on the dataframe."
        )

    # initialize the lists
    n_rows = len(df)
    mean_intensity_unique = np.zeros(n_rows)
    color_factor = np.zeros(n_rows)
    color_offset = np.zeros(n_rows)
    phase_label = []

    # loop over the rows of the dataframe
    for index, row in df.iterrows():
        ch3_norm = row[GeneratedColumns.MEAN_INTENSITY_CH3_NORM]
        ch4_norm = row[GeneratedColumns.MEAN_INTENSITY_CH4_NORM]

        # get fucci phase, and its start and end
        phase, start, end = FucciPhases.get_phase(ch3_norm, ch4_norm)
        phase_label.append(phase.value)

        # record factor and offset
        color_factor[index] = end - start
        color_offset[index] = start

        # record unique mean
        if phase == Phase.G1:
            mean_intensity_unique[index] = ch3_norm
        else:
            mean_intensity_unique[index] = ch4_norm

    # compute color intensity, rounded as an int vector
    color_intensity = np.rint(
        mean_intensity_unique * color_factor + color_offset,
    )
    color_str = [f"r={val};g={val};b={val};" for val in color_intensity]

    # update the dataframe
    df[GeneratedColumns.MEAN_INTENSITY_UNIQUE_VALUE] = mean_intensity_unique
    df[GeneratedColumns.PHASE] = phase
    df[GeneratedColumns.MANUAL_SPOT_COLOR] = color_str
