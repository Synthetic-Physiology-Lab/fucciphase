from fucciphase.io import read_trackmate_xml
from fucciphase.phase import NewColumns, compute_phase_trigo
from fucciphase.utils import normalize_channels, simulate_single_track


def test_smoke_pipeline_simulated():
    """Test that the pipeline can run on simulated data."""
    # simulate a single track
    df = simulate_single_track()

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"
    normalize_channels(df, [channel1, channel2])

    # compute the phase
    compute_phase_trigo(df, channel1, channel2)


def test_smoke_pipeline_trackmate(tmp_path, trackmate_xml):
    """Test that the pipeline can run on trackmate data."""
    # import the xml
    df, tmxml = read_trackmate_xml(trackmate_xml)

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH1"
    channel2 = "MEAN_INTENSITY_CH2"
    normalize_channels(df, [channel1, channel2])

    # compute the phase
    compute_phase_trigo(df, channel1, channel2)

    # update the XML
    tmxml.update_features(df)

    # export the XML
    path = tmp_path / "test.xml"
    tmxml.save_xml(path)

    # load it back and check that the new columns are there
    df2, _ = read_trackmate_xml(path)
    assert NewColumns.unified_intensity() in df2.columns
    assert NewColumns.color() in df2.columns
