import numpy as np
import pandas as pd
import pytest
from fucciphase.utils import moving_average, norm, normalize_channels


def test_norm():
    """Test the norm function for both numpy.array and pandas.Series."""
    v_min = 4
    v_max = 10

    # numpy
    vector = np.arange(v_min, v_max + 1)
    norm_vector = norm(vector)

    expected = np.round((vector - v_min) / (v_max - v_min), 2)
    assert (norm_vector == expected).all()

    # pandas
    df = pd.DataFrame({"vector": vector})
    norm_df = norm(df["vector"])

    expected_df = pd.DataFrame({"vector": expected})
    assert norm_df.equals(expected_df["vector"])


@pytest.mark.parametrize("use_ma", [True, False])
def test_normalize(trackmate_df: pd.DataFrame, use_ma):
    """Normalize the channels and test that the columns have
    been added to the dataframe."""

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"
    new_channels = normalize_channels(
        trackmate_df, [channel1, channel2], use_moving_average=use_ma
    )

    # check that the columns have been added
    channel1_norm = "MEAN_INTENSITY_CH3_NORM"
    channel2_norm = "MEAN_INTENSITY_CH4_NORM"
    assert channel1_norm in trackmate_df.columns and channel1_norm in new_channels
    assert channel2_norm in trackmate_df.columns and channel2_norm in new_channels

    # check if normalized
    assert trackmate_df[channel1_norm].min() == 0
    assert trackmate_df[channel1_norm].max() == 1
    assert trackmate_df[channel2_norm].min() == 0
    assert trackmate_df[channel2_norm].max() == 1


@pytest.mark.parametrize(
    "window, expected",
    [
        (3, [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 8.5]),
        (5, [1, 1.5, 2, 3, 4, 5, 6, 7, 7.5, 8]),
        (7, [1.5, 2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5]),
    ],
)
def test_moving_average(window, expected):
    """Test that the moving average function works as expected."""
    v = np.arange(0, 10)

    # expected value as a np array
    expected = np.array(expected)

    # run moving average
    result = moving_average(v, window)
    assert result.shape == expected.shape
    assert (result == expected).all()
