.. _CLI:

Command-line usage
==================

The ``fucciphase`` command accepts:

* TrackMate XML files
* tabular ``.csv`` files
* tabular ``.xlsx`` files

Small CSV/XLSX example
----------------------

.. code-block:: bash

    fucciphase examples/cli_quickstart/tiny_tracks.csv \
        -ref examples/cli_quickstart/tiny_reference.csv \
        --sensor_file examples/cli_quickstart/tiny_sensor.json \
        -dt 0.48 \
        -m MEAN_INTENSITY_CH4 \
        -c MEAN_INTENSITY_CH3

This writes ``outputs/tiny_tracks_processed.csv``.
An expected output table is provided in
``examples/cli_quickstart/tiny_tracks_expected_output.csv``.

TrackMate XML example
---------------------

.. code-block:: bash

    fucciphase examples/reproducibility/inputs/merged_linked.ome.xml \
        -ref examples/reproducibility/inputs/hacat_fucciphase_reference.csv \
        -dt 0.25 \
        -m MEAN_INTENSITY_CH1 \
        -c MEAN_INTENSITY_CH2 \
        --generate_unique_tracks

This writes:

* ``outputs/merged_linked.ome_processed.csv``
* ``outputs/merged_linked.ome_processed.xml``

Napari visualization
--------------------

.. code-block:: bash

    fucciphase-napari outputs/merged_linked.ome_processed.csv \
        examples/reproducibility/inputs/downscaled_hacat.ome.tif \
        -m 0 -c 1 -s 2 --pixel_size 0.544

Napari extras
-------------

Napari support is optional. Install with:

.. code-block:: bash

    pip install -e ".[napari]"
