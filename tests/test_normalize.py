import numpy as np
import pandas as pd
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


def test_normalize(trackmate_df: pd.DataFrame):
    """Normalize the channels and test that the columns have
    been added to the dataframe."""

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH1"
    channel2 = "MEAN_INTENSITY_CH2"
    new_channels = normalize_channels(trackmate_df, [channel1, channel2])

    # check that the columns have been added
    channel1_norm = "MEAN_INTENSITY_CH1_NORM"
    channel2_norm = "MEAN_INTENSITY_CH2_NORM"
    assert channel1_norm in trackmate_df.columns and channel1_norm in new_channels
    assert channel2_norm in trackmate_df.columns and channel2_norm in new_channels

    # check that the values are correct
    ch1 = trackmate_df[channel1_norm]
    norm_ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min())
    assert (norm_ch1 == trackmate_df[channel1_norm]).all()

    ch2 = trackmate_df[channel2_norm]
    norm_ch2 = (ch2 - ch2.min()) / (ch2.max() - ch2.min())
    assert (norm_ch2 == trackmate_df[channel2_norm]).all()
