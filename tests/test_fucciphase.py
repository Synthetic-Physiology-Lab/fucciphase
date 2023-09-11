from fucciphase.phase import compute_phase_trigo
from fucciphase.simulator import simulate_single_track
from fucciphase.utils import normalize_channels


def test_smoke_pipeline():
    """Test that the pipeline can run on simulated data."""
    # simulate a single track
    df = simulate_single_track()

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"
    normalize_channels(df, [channel1, channel2])

    # compute the phase
    compute_phase_trigo(df, channel1, channel2)
