import pandas as pd
import pytest
from fucciphase.fucci_phase import FucciPhases, normalize_channel


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
    channel1 = "MEAN_INTENSITY_CH1"
    channel2 = "MEAN_INTENSITY_CH2"
    normalize_channel(trackmate_df, channel1)
    normalize_channel(trackmate_df, channel2)

    # check that the columns have been added
    channel1_norm = "MEAN_INTENSITY_CH1_NORM"
    channel2_norm = "MEAN_INTENSITY_CH2_NORM"
    assert channel1_norm in trackmate_df.columns
    assert channel2_norm in trackmate_df.columns

    # check that the values are correct
    ch1 = trackmate_df[channel1_norm]
    norm_ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min())
    assert (norm_ch1 == trackmate_df[channel1_norm]).all()

    ch2 = trackmate_df[channel2_norm]
    norm_ch2 = (ch2 - ch2.min()) / (ch2.max() - ch2.min())
    assert (norm_ch2 == trackmate_df[channel2_norm]).all()


def test_compute_phase_color():
    """Test the compute_phase_color function."""
    # TODO
    pass
