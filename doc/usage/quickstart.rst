.. _Quickstart:

Quickstart
==========

Fucci phase currently supports loading a
[TrackMate](https://imagej.net/plugins/trackmate/) XML file:

.. code-block:: python

    from fucciphase import process_trackmate

    trackmate_xml = "path/to/trackmate.xml"
    channel1 = "MEAN_INTENSITY_CH3"
    channel2 = "MEAN_INTENSITY_CH4"

    df = process_trackmate(trackmate_xml, channel1, channel2)
    print(df["CELL_CYCLE_PERC"])
