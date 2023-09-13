from fucciphase.io import read_trackmate_xml
from fucciphase.phase import NewColumns, compute_cell_cycle, generate_cycle_phases
from fucciphase.utils import normalize_channels, simulate_single_track


def test_smoke_pipeline_simulated():
    """Test that the pipeline can run on simulated data."""
    # simulate a single track
    df = simulate_single_track()

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"
    normalize_channels(df, [channel1, channel2])

    # compute the cell cycle percentage
    compute_cell_cycle(df, channel1, channel2)
    assert NewColumns.cell_cycle() in df.columns
    assert NewColumns.color() in df.columns

    # compute the phases
    generate_cycle_phases(
        df,
        phases=["G1", "T", "S", "G2M"],  # one more phase than thresholds
        thresholds=[0.04, 0.44, 0.56],
    )


def test_smoke_pipeline_trackmate(tmp_path, trackmate_xml):
    """Test that the pipeline can run on trackmate data."""
    # import the xml
    df, tmxml = read_trackmate_xml(trackmate_xml)

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH1"
    channel2 = "MEAN_INTENSITY_CH2"
    normalize_channels(df, [channel1, channel2])

    # compute the cell cycle percentage
    compute_cell_cycle(df, channel1, channel2)

    # compute the phases
    generate_cycle_phases(
        df,
        phases=["G1", "T", "S", "G2M"],  # one more phase than thresholds
        thresholds=[0.04, 0.44, 0.56],
    )

    # update the XML
    tmxml.update_features(df)

    # export the XML
    path = tmp_path / "test.xml"
    tmxml.save_xml(path)

    # load it back and check that the new columns are there
    df2, _ = read_trackmate_xml(path)
    assert NewColumns.cell_cycle() in df2.columns
    assert NewColumns.color() in df2.columns
