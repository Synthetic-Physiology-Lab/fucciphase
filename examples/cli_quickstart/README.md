# CLI quickstart example

This folder contains a small reviewer-facing example for running FUCCIphase
without writing Python code.

## Files

- `tiny_tracks.csv`: small tabular input example
- `tiny_tracks.xlsx`: the same input saved as Excel
- `tiny_reference.csv`: reference curve used for percentage alignment
- `tiny_sensor.json`: sensor parameters used by the example command
- `tiny_tracks_expected_output.csv`: expected processed output table

## Run the CSV example

From the repository root:

```bash
fucciphase examples/cli_quickstart/tiny_tracks.csv \
    -ref examples/cli_quickstart/tiny_reference.csv \
    --sensor_file examples/cli_quickstart/tiny_sensor.json \
    -dt 0.48 \
    -m MEAN_INTENSITY_CH4 \
    -c MEAN_INTENSITY_CH3
```

This writes:

```text
outputs/tiny_tracks_processed.csv
```

Compare it against:

```text
examples/cli_quickstart/tiny_tracks_expected_output.csv
```

## Run the XLSX example

```bash
fucciphase examples/cli_quickstart/tiny_tracks.xlsx \
    -ref examples/cli_quickstart/tiny_reference.csv \
    --sensor_file examples/cli_quickstart/tiny_sensor.json \
    -dt 0.48 \
    -m MEAN_INTENSITY_CH4 \
    -c MEAN_INTENSITY_CH3
```

The small example assets are also configured to ship with the wheel under
`fucciphase/data/cli_quickstart/`.
