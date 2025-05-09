from fucciphase import process_dataframe, process_trackmate
from fucciphase.io import read_trackmate_xml
from fucciphase.phase import NewColumns, generate_cycle_phases
from fucciphase.sensor import FUCCISASensor
from fucciphase.utils import normalize_channels, simulate_single_track

sensor_dict = {
    "phase_percentages": [30.0, 37.0, 33.0],
    "center": [20, 55, 70, 99],
    "sigma": [7.2, 4.3, 13.0, 0.3],
}


def test_smoke_pipeline_simulated() -> None:
    """Test that the pipeline can run on simulated data."""
    # simulate a single track
    df = simulate_single_track()
    df2 = df.copy()

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"
    normalize_channels(df, [channel1, channel2], use_moving_average=True)

    sensor = FUCCISASensor(**sensor_dict)

    # compute the phases
    generate_cycle_phases(
        df,
        [channel1, channel2],
        sensor,
        thresholds=[0.1, 0.1],
    )

    # compare with high level API
    assert NewColumns.cell_cycle() not in df2.columns
    process_dataframe(
        df2,
        [channel1, channel2],
        sensor,
        thresholds=[0.1, 0.1],
        estimate_percentage=False,
    )
    assert df.equals(df2)


def test_smoke_pipeline_trackmate(tmp_path, trackmate_xml):
    """Test that the pipeline can run on trackmate data."""
    # import the xml
    df, tmxml = read_trackmate_xml(trackmate_xml)

    # normalize the channels
    channel1 = "MEAN_INTENSITY_CH1"
    channel2 = "MEAN_INTENSITY_CH2"
    normalize_channels(df, [channel1, channel2], use_moving_average=True)

    sensor = FUCCISASensor(**sensor_dict)

    # compute the phases
    generate_cycle_phases(
        df,
        [channel1, channel2],
        sensor,
        thresholds=[0.1, 0.1],
    )

    # update the XML
    tmxml.update_features(df)

    # export the XML
    path = tmp_path / "test.xml"
    tmxml.save_xml(path)

    # load it back and check that the new columns are there
    df2, _ = read_trackmate_xml(path)
    # check for phase only
    assert NewColumns.discrete_phase_max() in df2.columns

    # process it with high level API
    df3 = process_trackmate(
        trackmate_xml,
        [channel1, channel2],
        sensor,
        thresholds=[0.1, 0.1],
        estimate_percentage=False,
    )
    assert df.equals(df3)
