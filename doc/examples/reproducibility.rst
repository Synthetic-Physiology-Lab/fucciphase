.. _reproducibility:

Reproducibility example (TrackMate XML)
=======================================

A larger end-to-end workflow lives in ``examples/reproducibility``.

Committed inputs:

* ``inputs/merged_linked.ome.xml``
* ``inputs/hacat_fucciphase_reference.csv``
* ``inputs/downscaled_hacat.ome.tif``
* ``inputs/labels.tif``

Committed preview assets:

* ``outputs/thumbnail.png``
* ``outputs/video_downscaled_hacat.mp4``

Generated when you run the workflow:

* ``outputs/merged_linked.ome_processed.csv``
* ``outputs/merged_linked.ome_processed.xml``

Process the example XML:

.. code-block:: bash

    fucciphase inputs/merged_linked.ome.xml \
        -ref inputs/hacat_fucciphase_reference.csv \
        -dt 0.25 \
        -m MEAN_INTENSITY_CH1 \
        -c MEAN_INTENSITY_CH2 \
        --generate_unique_tracks

Visualize in Napari:

.. code-block:: bash

    fucciphase-napari outputs/merged_linked.ome_processed.csv \
        inputs/downscaled_hacat.ome.tif \
        -m 0 -c 1 -s 2 --pixel_size 0.544

The TrackMate XML stores an absolute ``folder`` path inside ``<ImageData>``.
FUCCIphase ignores that path during processing.
