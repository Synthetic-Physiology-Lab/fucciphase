import pytest
from fucciphase.phase import FucciColorIndex, _get_phase_bichannel


def test_fucci_phases():
    """Test that the Fucci phases cover 256 values."""
    tot = (
        FucciColorIndex.end_early_G1
        - FucciColorIndex.start_early_G1
        + 1
        + FucciColorIndex.end_G1
        - FucciColorIndex.start_G1
        + 1
        + FucciColorIndex.end_S
        - FucciColorIndex.start_S
        + 1
        + FucciColorIndex.end_G2_M
        - FucciColorIndex.start_G2_M
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
    fucci_phase, _, _ = _get_phase_bichannel(ch3, ch4)
    assert fucci_phase == phase


def test_get_phase_bichannel():
    # TODO
    # - create synthetic data with expected result
    # - run function
    # - check results
    pass
