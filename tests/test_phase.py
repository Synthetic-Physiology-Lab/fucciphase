import numpy as np
import pandas as pd
import pytest
from fucciphase.phase import NewColumns, compute_cell_cycle
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
