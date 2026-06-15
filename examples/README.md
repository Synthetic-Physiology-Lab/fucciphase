# FUCCIphase examples

This folder contains the main example and tutorial material for FUCCIphase.

## Suggested starting points

1. `cli_quickstart/`
   Small reviewer-facing CLI example with CSV, XLSX, reference data, sensor
   parameters, and an expected output table.
2. `reproducibility/`
   Larger TrackMate XML workflow with Napari visualization and preview media.
3. `notebooks/`
   Jupyter notebooks for calibration, reconstruction, simulation, and figure
   generation.
4. `example_data/`
   Reference curves and saved sensor JSON files.

## Quick CLI example

From the repository root:

```bash
fucciphase examples/cli_quickstart/tiny_tracks.csv \
    -ref examples/cli_quickstart/tiny_reference.csv \
    --sensor_file examples/cli_quickstart/tiny_sensor.json \
    -dt 0.48 \
    -m MEAN_INTENSITY_CH4 \
    -c MEAN_INTENSITY_CH3
```

Expected output:

```text
outputs/tiny_tracks_processed.csv
```

Reference table for comparison:

```text
examples/cli_quickstart/tiny_tracks_expected_output.csv
```

The same example is also provided as:

```text
examples/cli_quickstart/tiny_tracks.xlsx
```

## Full TrackMate reproducibility workflow

The TrackMate XML example in `reproducibility/` uses:

- `inputs/merged_linked.ome.xml`
- `inputs/hacat_fucciphase_reference.csv`
- `inputs/downscaled_hacat.ome.tif`

Run it from `examples/reproducibility/`:

```bash
fucciphase inputs/merged_linked.ome.xml \
    -ref inputs/hacat_fucciphase_reference.csv \
    -dt 0.25 \
    -m MEAN_INTENSITY_CH1 \
    -c MEAN_INTENSITY_CH2 \
    --generate_unique_tracks
```

This generates:

```text
outputs/merged_linked.ome_processed.csv
outputs/merged_linked.ome_processed.xml
```

Preview assets are committed in the repository. The processed CSV/XML files are
generated when you run the workflow.

## Napari visualization

```bash
fucciphase-napari outputs/merged_linked.ome_processed.csv \
    inputs/downscaled_hacat.ome.tif \
    -m 0 -c 1 -s 2 --pixel_size 0.544
```

## Support

Report bugs or feature requests at:
https://github.com/Synthetic-Physiology-Lab/fucciphase/issues
