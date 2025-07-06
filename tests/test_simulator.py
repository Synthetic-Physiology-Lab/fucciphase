import numpy as np

from fucciphase.utils.simulator import simulate_single_channel, simulate_single_track


def test_simulate_single_channel():
    # create time array
    t = np.arange(0, 50)

    mean = 25
    sigma = 10

    # simulate a channel
    ch = simulate_single_channel(t, mean, sigma)

    # check that shape is correct
    assert ch.shape == t.shape

    # check that the min and max are different
    assert ch.min() != ch.max()

    # check that the peak corresponds to about the mean position (rounding errors)
    assert np.abs(ch.argmax() - mean) <= 1


def test_simulate_single_track():
    # get single track
    df = simulate_single_track()

    # check that the mean intensity channel 3 and 4 are there
    assert "MEAN_INTENSITY_CH3" in df.columns
    assert "MEAN_INTENSITY_CH4" in df.columns

    # ... and that they have different mean, min and max
    assert df["MEAN_INTENSITY_CH3"].mean() != df["MEAN_INTENSITY_CH4"].mean()
    assert df["MEAN_INTENSITY_CH3"].min() != df["MEAN_INTENSITY_CH4"].min()
    assert df["MEAN_INTENSITY_CH3"].max() != df["MEAN_INTENSITY_CH4"].max()
