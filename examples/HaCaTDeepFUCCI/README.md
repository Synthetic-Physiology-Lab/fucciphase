# HaCaT DeepFUCCI Analysis Pipeline

This example demonstrates the complete FUCCIphase analysis pipeline for estimating cell cycle percentages from FUCCI imaging data using DTW (Dynamic Time Warping) subsequence alignment.

## Overview

The pipeline consists of two main steps:

1. **Compute**: Process TrackMate tracking data to estimate cell cycle percentages
2. **Visualize**: Display results in napari with tracks colored by cell cycle phase

## Input Data

| File | Description |
|------|-------------|
| `2.nd2` | Raw microscopy image (ND2 format, multi-channel time-lapse) |
| `downscaled_hacat_100x.ome.tif` | Downscaled image (9x smaller, OME-TIFF format) |
| `merged.ome.xml` | TrackMate tracking data (XML format) |
| `stardist_labels_3_channel.tif` | Segmentation labels from StarDist |
| `../example_data/fuccisa_hacat.json` | Sensor configuration for HaCaT cells |
| `../example_data/hacat_fucciphase_reference.csv` | Reference curve for DTW alignment |

## Step 1: Compute Cell Cycle Percentages

```bash
python compute_fucciphase.py \
    --track-file merged.ome.xml \
    --sensor-file ../example_data/fuccisa_hacat.json \
    --reference-file ../example_data/hacat_fucciphase_reference.csv \
    --output processed_tracks.csv \
    --signal-mode both \
    --signal-weight 1.0 \
    --signal-smooth 11 \
    --penalty 0.05 \
    --plot-figures \
    --figures-dir figures
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--signal-mode` | DTW feature mode: `signal`, `derivative`, or `both` | `both` |
| `--signal-weight` | Weight for signal vs derivative in "both" mode (1.0 = equal) | `1.0` |
| `--signal-smooth` | Savitzky-Golay smoothing window for signal (0 = none, must be > 3) | `11` |
| `--penalty` | DTW penalty for non-diagonal warping (higher = more linear) | `0.05` |
| `--min-length` | Minimum track length in frames | `12` |
| `--plot-figures` | Generate reference vs query comparison plots | - |

### Signal Modes

- **`derivative`** (default): Uses only the derivative of the signal. Baseline-independent but sensitive to noise.
- **`signal`**: Uses only the raw signal. Sensitive to baseline variations.
- **`both`**: Uses both signal and derivative as features. Best of both worlds when properly tuned.

### Output Columns

The output CSV contains the original tracking data plus:

| Column | Description |
|--------|-------------|
| `CELL_CYCLE_PERC_DTW` | Estimated cell cycle percentage (0-100%) |
| `DTW_DISTANCE` | DTW alignment distance (lower = better match) |
| `DTW_DISTORTION` | Time distortion in the alignment |
| `DTW_WARP` | Warping amount (deviation from diagonal) |

## Step 2: Visualize in Napari

### Using full-resolution ND2 file

```bash
python visualize_napari.py \
    --tracks processed_tracks.csv \
    --image 2.nd2 \
    --labels stardist_labels_3_channel.tif
```

### Using downscaled OME-TIFF (faster loading)

For faster visualization, use the 9x downscaled OME-TIFF file:

```bash
python visualize_napari_ometiff.py \
    --tracks processed_tracks.csv \
    --image downscaled_hacat_100x.ome.tif
```

The downscaled image is 9x smaller than the original. Track coordinates are automatically scaled by 1/9 to match.

### Screenshot Generation

To generate screenshots at specific frames:

```bash
python visualize_napari.py \
    --tracks processed_tracks.csv \
    --image 2.nd2 \
    --labels stardist_labels_3_channel.tif \
    --screenshots 0 13 26 39 \
    --screenshot-dir screenshots/output \
    --screenshot-prefix frame \
    --no-gui
```

### Video Generation

To create an animation:

```bash
python visualize_napari.py \
    --tracks processed_tracks.csv \
    --image 2.nd2 \
    --output video.mp4 \
    --fps 4
```

## Directory Structure

```
HaCaTDeepFUCCI/
├── README.md                      # This file
├── compute_fucciphase.py          # Cell cycle computation script
├── visualize_napari.py            # Napari visualization (ND2)
├── visualize_napari_ometiff.py    # Napari visualization (OME-TIFF, faster)
├── 2.nd2                          # Raw image data
├── downscaled_hacat_100x.ome.tif  # 9x downscaled image (faster loading)
├── merged.ome.xml                 # TrackMate tracking data
├── stardist_labels_3_channel.tif  # Segmentation labels
├── processed_tracks_final.csv     # Final processed output
├── figures_*/                     # Comparison plots (reference vs query)
└── screenshots/                   # Generated screenshots
    └── README.md                  # Screenshot documentation
```

## Recommended Workflow

1. **Initial run** with default parameters to assess data quality:
   ```bash
   python compute_fucciphase.py --plot-figures --figures-dir figures_initial
   ```

2. **Review figures** to check alignment quality (reference vs query curves)

3. **Tune parameters** if needed:
   - Increase `--penalty` if tracks show unrealistic time jumps
   - Use `--signal-mode both` with `--signal-smooth 11` for noisy data
   - Adjust `--signal-weight` to balance signal vs derivative contribution

4. **Filter output** if needed (e.g., remove low-quality tracks based on DTW metrics)

5. **Visualize** final results in napari

## Citation

If you use this pipeline, please cite the FUCCIphase package.
