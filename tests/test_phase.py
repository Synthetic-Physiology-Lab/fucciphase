import numpy as np
import pandas as pd
import pytest
from fucciphase.phase import NewColumns, compute_cell_cycle, generate_cycle_phases
from fucciphase.utils import get_norm_channel_name


def test_no_normalised_channel():
    """Test than an error is raised when one of the normed channel is not present."""
    # create dataframe
    ch1 = "CH1"
    ch2 = "CH2"
    df = pd.DataFrame(
        columns=[ch1, ch2, "Something", "Else", get_norm_channel_name(ch2)]
    )

    # check that the error is raised
    with pytest.raises(ValueError):
        compute_cell_cycle(df, ch1, ch2)


def test_compute_cell_cycle():
    """Test that the cell cycle computation works."""
    # channel names
    channel1 = "ANY NAME"
    channel2 = "ANOTHER ONE"

    # create two nromalizedchannels
    ramp_up = np.arange(0, 25) / 25.0
    ramp_down = 1 - np.arange(0, 25) / 25.0

    ch1 = np.concatenate([ramp_up, ramp_down])
    ch2 = np.concatenate([ramp_down, ramp_up])

    # create dataframe
    df = pd.DataFrame(
        {get_norm_channel_name(channel1): ch1, get_norm_channel_name(channel2): ch2}
    )

    # compute cell cycle
    compute_cell_cycle(df, channel1, channel2)
    assert NewColumns.cell_cycle() in df.columns
    assert NewColumns.color() in df.columns

    # check that the cell cycle is between 0 and 1 and that its highest and lowest
    # values correspond to those of the second channel
    cell_cycle = df[NewColumns.cell_cycle()]
    assert cell_cycle.min() == 0
    assert cell_cycle.max() == 1
    assert cell_cycle.argmin() == ch2.argmin()
    assert cell_cycle[:25].argmax() == ch2[:25].argmax()
    assert cell_cycle[25:].argmax() == ch2[25:].argmax()

    # check the color
    # TODO how to test that the color scheme is correct?


@pytest.mark.parametrize(
    "phase, thresholds",
    [
        # different number of elements than expected
        (["S", "G1"], [0.1, 0.4, 0.2, 0.8]),
        (["M", "S", "G1"], [0.4]),
        # duplicated elements
        (["M", "S", "G1", "M"], [0.4, 0.2, 0.8]),
        (["M", "S", "G1"], [0.4, 0.4]),
        # not between 0 and 1
        (["M", "S", "G1", "T"], [-0.1, 0.4, 0.8]),
        (["M", "S", "G1", "T"], [0.1, 0.4, 1]),
    ],
)
def test_generate_cycle_phases_errors(phase, thresholds):
    """Test that errors are raised"""
    # create dataframe
    df = pd.DataFrame(
        {
            NewColumns.cell_cycle(): np.arange(0, 10) / 10,
        }
    )

    # check that the error is raised
    with pytest.raises(ValueError):
        generate_cycle_phases(df, phase, thresholds)


def test_generate_phases_no_cell_cycle():
    """Test than an error is raised when the cell cycle column is missing."""
    df = pd.DataFrame(
        {
            "Cell": np.arange(0, 10) / 10,
        }
    )

    # get phase and thresholds
    phases = ["S", "G1", "T", "G2M"]
    thresholds = [0.4, 0.04, 0.56]

    with pytest.raises(ValueError):
        generate_cycle_phases(df, phases, thresholds)


def test_generate_phases():
    """Test that the phases are correctly attributed."""
    # create data
    ramp_up = np.arange(0, 10) / 10.0
    ramp_down = 1 - np.arange(0, 10) / 10.0
    df = pd.DataFrame(
        {
            NewColumns.cell_cycle(): np.concatenate([ramp_up, ramp_down]),
        }
    )

    # create phases and thresholds
    phases = ["S", "T", "G1", "G2M"]
    thresholds = [0.81, 0.41, 0.11]

    # expected result
    phase_vector = ["G1" for _ in range(2)]
    phase_vector.extend(["T" for _ in range(3)])
    phase_vector.extend(["S" for _ in range(4)])
    phase_vector.extend(["G2M" for _ in range(1)])
    phase_vector.extend(["G2M" for _ in range(2)])
    phase_vector.extend(["S" for _ in range(4)])
    phase_vector.extend(["T" for _ in range(3)])
    phase_vector.extend(["G1" for _ in range(1)])

    # generate phases
    generate_cycle_phases(df, phases, thresholds)

    # check that the phases are correct
    assert all(df[NewColumns.phase()] == np.array(phase_vector))
