.. _cli_quickstart:

CLI quickstart example
======================

The folder ``examples/cli_quickstart`` contains a small reviewer-facing example
for command-line use without writing Python code.

It includes:

* ``tiny_tracks.csv``
* ``tiny_tracks.xlsx``
* ``tiny_reference.csv``
* ``tiny_sensor.json``
* ``tiny_tracks_expected_output.csv``

Run the CSV example from the repository root:

.. code-block:: bash

    fucciphase examples/cli_quickstart/tiny_tracks.csv \
        -ref examples/cli_quickstart/tiny_reference.csv \
        --sensor_file examples/cli_quickstart/tiny_sensor.json \
        -dt 0.48 \
        -m MEAN_INTENSITY_CH4 \
        -c MEAN_INTENSITY_CH3

This writes ``outputs/tiny_tracks_processed.csv``.
Compare it against ``tiny_tracks_expected_output.csv``.
