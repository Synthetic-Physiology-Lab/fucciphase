# TrackMate reproducibility workflow

This folder contains a larger end-to-end example showing how to run
FUCCIphase on a TrackMate XML file and visualize the result in Napari.

It is intended as a secondary workflow after the smaller
`examples/cli_quickstart/` example.

## Folder contents

Committed inputs:

- `inputs/merged_linked.ome.xml`
- `inputs/hacat_fucciphase_reference.csv`
- `inputs/downscaled_hacat.ome.tif`
- `inputs/labels.tif`

Committed preview assets:

- `outputs/thumbnail.png`
- `outputs/video_downscaled_hacat.mp4`

Generated when you run the tutorial:

- `outputs/merged_linked.ome_processed.csv`
- `outputs/merged_linked.ome_processed.xml`

## Run the example

From this folder:

```bash
fucciphase inputs/merged_linked.ome.xml \
    -ref inputs/hacat_fucciphase_reference.csv \
    -dt 0.25 \
    -m MEAN_INTENSITY_CH1 \
    -c MEAN_INTENSITY_CH2 \
    --generate_unique_tracks
```

This writes:

```text
outputs/merged_linked.ome_processed.csv
outputs/merged_linked.ome_processed.xml
```

For command help:

```bash
fucciphase -h
```

## Visualize in Napari

The TrackMate XML stores an absolute `folder` path inside `<ImageData>`.
FUCCIphase ignores this field during processing, so you do not need to edit it
to run this example.

Remember to install the visualization extra with `pip install -e ".[napari]"`.

```bash
fucciphase-napari outputs/merged_linked.ome_processed.csv \
    inputs/downscaled_hacat.ome.tif \
    -m 0 -c 1 -s 2 --pixel_size 0.544
```

## Optional notebooks

The related notebooks are in `../notebooks/`:

- `extract_calibration_data.ipynb`
- `percentage_reconstruction.ipynb`
- `phaselocking-workflow-lazy.ipynb`

## Preview

[![Preview of the video](outputs/thumbnail.png)](outputs/video_downscaled_hacat.mp4)

## Support

Report bugs or feature requests at:
https://github.com/Synthetic-Physiology-Lab/fucciphase/issues
