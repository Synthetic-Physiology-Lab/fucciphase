# How to use FUCCIphase from the command line 

This folder contains a complete, end-to-end example showing how to run FUCCIphase 
from the command line, no coding required.

The example shows how to:

1. Process a TrackMate XML file with the FUCCIphase command-line interface (CLI)
2. Estimate cell-cycle percentages using DTW subsequence alignment
3. Visualize results with Napari
4. Reproduce figures using Jupyter notebooks

The tutorial uses real FUCCI data and demonstrates the workflow described in the repository.

---
## Folder Structure

```
reproducibility/
│
├── inputs/          # Raw input data
│   ├── merged_linked.ome.xml          # TrackMate export with tracks
│   ├── downscaled_hacat.ome.tif       # Example fluorescence video (downscaled)
│   └── hacat_fucciphase_reference.csv # Reference FUCCI calibration curve
│
├── outputs/         
    ├── merged_linked.ome.xml_processed.csv  # Results generated during the tutorial
    ├── merged_linked.ome_processed.xml      # Results generated during the tutorial
    ├── thumbnail.png
    └── video_downscaled_hacat.mp4
```

---

# 1. Process the TrackMate XML file

This example uses the XML file at `inputs/merged_linked.ome.xml`. Make sure you are inside the `reproducibility folder` before running the command:
```bash
fucciphase inputs/merged_linked.ome.xml `
          -ref ../example_data/hacat_fucciphase_reference.csv `
          -dt 0.25 `
          -m MEAN_INTENSITY_CH1 `
          -c MEAN_INTENSITY_CH2 `
          --generate_unique_tracks true `
```

This will generate two files in the `outputs/` folder:


* outputs / merged_linked.ome.xml_processed.csv
* outputs / merged_linked.ome_processed.xml


These contain:

* normalized intensities
* discrete phase assignments
* DTW-based cell-cycle percentages
* updated TrackMate XML with new feature columns

For help on available options:

```bash
fucciphase -h
```

---

# 2. Visualize the results in Napari

We provide a downscaled OME-TIFF video to make the example easy to run on laptops:

```
examples/reproducibility/inputs/downscaled_hacat.ome.tif
```

Launch Napari:

```bash
fucciphase-napari examples/reproducibility/outputs/merged_linked.ome_processed.csv examples/reproducibility/inputs/downscaled_hacat.ome.tif -m 0 -c 1 -s 2 --pixel_size 0.544
```

Napari will display:

* fluorescence channels
* segmentation mask
* track overlays
* cell-cycle percentages as text labels

You can adjust:

* segmentation boundaries
* colormaps
* opacity
* text size

To create animations, consider:

📦 [https://napari.org/napari-animation/](https://napari.org/napari-animation/)

---

# 3. Example Output

Below is a thumbnail preview that links to the example video stored in `outputs/`:

[![Preview of the video](outputs/thumbnail.png)](outputs/video_downscaled_hacat.mp4)

---

# 4. Reproduce figures using notebooks (optional)

The `notebooks/` folder contains Jupyter notebooks that reproduce:

* [extract_calibration_data.ipynb](../notebooks/extract_calibration_data.ipynb)
* [percentage_reconstruction.ipynb](../notebooks/percentage_reconstruction.ipynb) to estimate the cell cycle percentages
* [phaselocking-workflow-lazy.ipynb](../notebooks/phaselocking-workflow-lazy.ipynb) to process a trackmate file and render a video in Napari

---

# 5. Use the workflow with your own data

To apply the pipeline to your own dataset:

1. Export a TrackMate XML file containing tracks
2. Build a reference FUCCI curve (see `extract_calibration_data.ipynb`)
3. Run FUCCIphase on your XML file:

   ```bash
   fucciphase your_tracks.xml -ref your_reference.csv -dt <hours> -m <ch1> -c <ch2>
   ```
4. Visualize in Napari:

   ```bash
   fucciphase-napari your_tracks_processed.csv your_video.tif -m <ch index> -c <ch index> -s <mask index>
   ```

---

# 6. Support

If you encounter issues or have feature requests, please open an issue:

[https://github.com/Synthetic-Physiology-Lab/fucciphase/issues](https://github.com/Synthetic-Physiology-Lab/fucciphase/issues)
