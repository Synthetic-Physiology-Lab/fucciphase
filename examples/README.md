# FUCCIphase – Examples and Tutorials

This folder contains practical examples showing how to use FUCCIphase for:

- processing TrackMate XML files  
- estimating cell-cycle phases and percentages  
- visualizing results with Napari  
- reproducing analysis pipelines using Jupyter notebooks  

The folder is organized into two main sections:
````
examples/
│
├── README.md
│
├── example_data/
│   ├── example_data_tempophase.csv
│   ├── fuccisa_hacat.json
│   ├── fuccisa_hela_molcel_2017.json
│   ├── fuccisa_tempophase.json
│   └── hacat_fucciphase_reference.csv
│
├── notebooks/
│   ├── color-tails-by-percentage.ipynb
│   ├── example_estimated.ipynb
│   ├── example_reconstruction.ipynb
│   ├── example_simulated.ipynb
│   ├── explanation-dtw-alignment.ipynb
│   ├── extract_calibration_data.ipynb
│   ├── getting_started.ipynb
│   ├── movie_tails_percentages.mp4
│   ├── percentage_reconstruction.ipynb
│   ├── phaselocking-workflow-lazy.ipynb
│   ├── sensor_calibration.ipynb
│   └── README.md
│
└── reproducibility/
    ├── README.md
    │
    ├── inputs/
    │   ├── downscaled_hacat.ome.tif
    │   ├── hacat_fucciphase_reference.csv
    │   └── merged_linked.ome.xml
    │
    └── outputs/
        ├── thumbnail.png
        └── video_downscaled_hacat.mp4
````
---
# 1. Quickstart: run FUCCIphase on your data

If you already have:

   - a TrackMate XML file  
   - a FUCCI reference CSV  
   - your imaging timestep
      
you can run FUCCIphase from the command line:

```bash
fucciphase path/to/your_tracks.xml -ref path/to/your_reference.csv -dt 0.25 -m MEAN_INTENSITY_CH1 -c MEAN_INTENSITY_CH2 --generate_unique_tracks true
````

This produces a processed CSV:
```
your_tracks.xml_processed.csv
```
containing:
* normalized intensities
* discrete phases
* DTW-based cell-cycle percentages
* per-track metadata

For more details:
```bash
fucciphase -h
```

---

# 2. Visualize your results in Napari

You can launch the Napari viewer with:

```bash
fucciphase-napari your_tracks_processed.csv your_video.ome.tif -m 0 -c 1 -s 2 --pixel_size <pixel size>
```

This opens:

* raw channels
* segmentation masks
* track overlays
* estimated percentages as floating labels

Useful for:

* debugging
* figure creation
* validating results visually

---

# 3. Example Jupyter notebooks

The `notebooks/` folder contains lightweight notebooks demonstrating:

| Notebook                        | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| getting_started.ipynb           | Minimal example for new users                   |
| example_simulated.ipynb         | Simulated two-channel FUCCI traces              |
| example_estimated.ipynb         | Percentage estimation walkthrough               |
| example_reconstruction.ipynb    | Reconstruct intensity profiles from percentages |
| explanation-dtw-alignment.ipynb | How DTW subsequence alignment works             |
| color-tails-by-percentage.ipynb | Example trajectory coloring                     |
| sensor_calibration.ipynb        | Building reference traces                       |

These notebooks are intended as **mini-tutorials** for common tasks.

---

# 4. Full reproducibility tutorial

A complete, end-to-end workflow with real data is located in:

[reproducibility folder](./reproducibility/)

It includes:

* **inputs**: TrackMate XML, reference CSV, example video
* **outputs**: processed CSV files, thumbnails, exported XML

This is the recommended starting point for reproducing the figures shown in the repository.

---
# 5. Troubleshooting & Support

If you find bugs, unexpected behavior, or want new features, open an issue:

🔗 [https://github.com/Synthetic-Physiology-Lab/fucciphase/issues](https://github.com/Synthetic-Physiology-Lab/fucciphase/issues)