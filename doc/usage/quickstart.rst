.. _Quickstart:

Quickstart
==========

Command-line quickstart
-----------------------

The smallest runnable example is in ``examples/cli_quickstart``.
Run it from the repository root:

.. code-block:: bash

    fucciphase examples/cli_quickstart/tiny_tracks.csv \
        -ref examples/cli_quickstart/tiny_reference.csv \
        --sensor_file examples/cli_quickstart/tiny_sensor.json \
        -dt 0.48 \
        -m MEAN_INTENSITY_CH4 \
        -c MEAN_INTENSITY_CH3

This writes ``outputs/tiny_tracks_processed.csv``.
Compare the result against
``examples/cli_quickstart/tiny_tracks_expected_output.csv``.

The same example is also shipped as ``tiny_tracks.xlsx``.

TrackMate XML workflow
----------------------

For a larger TrackMate XML example with Napari visualization, see
``examples/reproducibility``.

Python API
----------

You can also call the pipeline directly:

.. code-block:: python

    from fucciphase import process_trackmate
    from fucciphase.sensor import FUCCISASensor

    sensor = FUCCISASensor(
        phase_percentages=[33.3, 33.3, 33.4],
        center=[20.0, 55.0, 70.0, 95.0],
        sigma=[5.0, 5.0, 10.0, 1.0],
    )

    df = process_trackmate(
        "path/to/trackmate.xml",
        channels=["MEAN_INTENSITY_CH1", "MEAN_INTENSITY_CH2"],
        sensor=sensor,
        thresholds=[0.1, 0.1],
    )
    print(df[["TRACK_ID", "CELL_CYCLE_PERC_DTW"]].head())
