import pandas as pd
import pytest
from fucciphase.fucci_phase import FucciPhases, normalize_channels


def test_fucci_phases():
    """Test that the Fucci phases cover 256 values."""
    tot = (
        FucciPhases.end_early_G1
        - FucciPhases.start_early_G1
        + 1
        + FucciPhases.end_G1
        - FucciPhases.start_G1
        + 1
        + FucciPhases.end_S
        - FucciPhases.start_S
        + 1
        + FucciPhases.end_G2_M
        - FucciPhases.start_G2_M
        + 1
    )
    assert tot == 256


@pytest.mark.parametrize(
    "ch3, ch4, phase",
    [
        (0.01, 0.02, "EG1"),
        (0.1, 0.1, "EG1"),
        (0.01, 0.11, "G1"),
        (0.11, 0.11, "T"),
        (0.11, 0.05, "G2/M"),
    ],
)
def test_get_phase(ch3: float, ch4: float, phase: str):
    """Test the get_phase function."""
    fucci_phase, _, _ = FucciPhases.get_phase(ch3, ch4)
    assert fucci_phase == phase


def test_normalize(trackmate_df: pd.DataFrame):
    """Normalize the channels and test that the columns have
    been added to the dataframe."""

    # normalize the channels
    normalize_channels(trackmate_df)

    # check that the columns have been added
    assert "MEAN_INTENSITY_CH3_NORM" in trackmate_df.columns
    assert "MEAN_INTENSITY_CH4_NORM" in trackmate_df.columns

    # check that the values are correct
    ch3 = trackmate_df["MEAN_INTENSITY_CH3_NORM"]
    norm_ch3 = (ch3 - ch3.min()) / (ch3.max() - ch3.min())
    assert (norm_ch3 == trackmate_df["MEAN_INTENSITY_CH3_NORM"]).all()

    ch4 = trackmate_df["MEAN_INTENSITY_CH4_NORM"]
    norm_ch4 = (ch4 - ch4.min()) / (ch4.max() - ch4.min())
    assert (norm_ch4 == trackmate_df["MEAN_INTENSITY_CH4_NORM"]).all()
