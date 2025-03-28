import numpy as np
import pandas as pd
import pytest

from fucciphase.utils import norm, normalize_channels


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
    # test only without moving average (enough on raw data to ensure correctness)
    if not use_ma:
        idx_min_channel1 = trackmate_df[channel2].argmax()
        idx_min_channel2 = trackmate_df[channel1].argmax()
        min_channel1 = trackmate_df[channel1][idx_min_channel1]
        min_channel2 = trackmate_df[channel2][idx_min_channel2]
        channel1_norm_expected = (trackmate_df[channel1] - min_channel1) / (
            trackmate_df[channel1].max() - min_channel1
        )
        channel1_norm_expected = np.clip(channel1_norm_expected, 0, 100)
        channel1_norm_expected = np.round(channel1_norm_expected, decimals=2)
        channel2_norm_expected = (trackmate_df[channel2] - min_channel2) / (
            trackmate_df[channel2].max() - min_channel2
        )
        channel2_norm_expected = np.clip(channel2_norm_expected, 0, 100)
        channel2_norm_expected = np.round(channel2_norm_expected, decimals=2)
        assert trackmate_df[channel1_norm].min() == channel1_norm_expected.min()
        assert trackmate_df[channel1_norm].max() == channel1_norm_expected.max()
        assert trackmate_df[channel2_norm].min() == channel2_norm_expected.min()
        assert trackmate_df[channel2_norm].max() == channel2_norm_expected.max()


def test_normalize_manual_minmax_wrong_list(trackmate_df: pd.DataFrame):
    """Test that manual normalization raises an error if not enough min/max values"""
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"

    # get min, max, for both channels
    min_ch = [trackmate_df[channel1].min()]
    max_ch = [trackmate_df[channel1].max()]

    # test error
    with pytest.raises(ValueError):
        normalize_channels(
            trackmate_df, [channel1, channel2], manual_min=min_ch, manual_max=max_ch
        )


def test_normalize_manual_minmax(trackmate_df: pd.DataFrame):
    """Test manual normalization"""
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"

    # get min, max, for both channels
    max_idx_ch1 = trackmate_df[channel1].argmax()
    max_idx_ch2 = trackmate_df[channel2].argmax()
    min_ch = [trackmate_df[channel1][max_idx_ch2], trackmate_df[channel2][max_idx_ch1]]
    max_ch = [trackmate_df[channel1].max(), trackmate_df[channel2].max()]

    # duplicate dataframe
    df = trackmate_df.copy()

    # normalize
    _ = normalize_channels(trackmate_df, [channel1, channel2], use_moving_average=False)

    # normalize by passing the values
    _ = normalize_channels(
        df,
        [channel1, channel2],
        manual_min=min_ch,
        manual_max=max_ch,
        use_moving_average=False,
    )

    # check that the values are the same
    assert df.equals(trackmate_df)
