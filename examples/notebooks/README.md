# FUCCIphase example notebooks

This folder contains a set of Jupyter notebooks that demonstrate how to use
**FUCCIphase** in different scenarios, from basic usage to more advanced
workflows such as calibration, simulation and reconstruction.

All notebooks assume that:

- `fucciphase` is installed in your environment (e.g. `pip install -e .` from the repo root)
- `jupyter` is installed (e.g. `pip install jupyter`)
- You run the notebooks from the **repository root** or from within
  the `examples` folder so that relative paths to data files work.

> 💡 Tip: start Jupyter from the repo root:
> ```bash
> cd path/to/fucciphase
> jupyter notebook
> ```
> then navigate to `examples/notebooks/`.

## Notebook overview

| Notebook file                          | Purpose                                                                                              | 
|----------------------------------------|------------------------------------------------------------------------------------------------------|
| `getting_started.ipynb`                | Minimal end-to-end example: load TrackMate output, apply a FUCCI sensor, inspect results.            | 
| `extract_calibration_data.ipynb`       | Extract FUCCI calibration time-courses from raw movies (TrackMate XML + images) to build references. | 
| `sensor_calibration.ipynb`             | Build or inspect a FUCCI sensor from calibration data (e.g. HaCaT reference traces).                 | 
| `explanation-dtw-alignment.ipynb`      | Explain and demonstrate the DTW-based alignment used for phase estimation and CALIPERS workflows.    | 
| `example_simulated.ipynb`              | Simulate synthetic FUCCI trajectories and test FUCCIphase on controlled, ground-truth data.          |
| `example_reconstruction.ipynb`         | Reconstruct FUCCI signals or phase trajectories from partial / noisy data and compare to originals.  | 
| `percentage_reconstruction.ipynb`      | Reconstruct and smooth phase-percentage trajectories and compare raw vs reconstructed dynamics.      | 
| `phaselocking-workflow-lazy.ipynb`     | Phase-lock and align many trajectories to a common reference using a lightweight, scalable workflow. | 
| `example_estimated.ipynb`              | Explore and visualise FUCCIphase output tables with per-cell phase estimates over time.              | 
| `color-tails-by-percentage.ipynb`      | Visualise phase distributions over time or conditions as “colour tail” plots for figures.            |