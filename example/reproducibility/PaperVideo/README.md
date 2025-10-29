Process a TrackMate file through the command-line interface (CLI):

```
fucciphase merged_linked.ome.xml -ref ../../example_data/hacat_fucciphase_reference.csv -dt 0.25 -m MEAN_INTENSITY_CH1 -c MEAN_INTENSITY_CH2 --generate_unique_tracks true
```

To get more info, run
```
fucciphase -h
```

Visualize the file from the CLI:

We provide a downscaled version of the video for faster processing:

```
fucciphase-napari merged_linked.ome.xml_processed.csv downscaled_hacat.ome.tif -m 0 -c 1 -s 2 --pixel_size 0.544
```

Please adjust segmentation contours etc. in napari.
Consider [napari-animate](https://napari.org/napari-animation/) to prepare animations.

Click on the thumbnail for an example:
[![Preview of the video](https://raw.githubusercontent.com/Synthetic-Physiology-Lab/fucciphase/cli/example/reproducibility/PaperVideo/thumbnail.png)](https://raw.githubusercontent.com/Synthetic-Physiology-Lab/fucciphase/cli/example/reproducibility/PaperVideo/video_downscaled_hacat.mp4)

