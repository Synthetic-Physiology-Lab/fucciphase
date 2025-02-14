# Examples

1. `example_simulated.ipynb`:
   We show exemplarily, how simulated sensor data gets processed.

1. `getting_started.ipynb`:
   First notebook to get into the fucciphase workflow.
   Requires a calibrated sensor.
    
1. `sensor_calibration.ipynb`: 
   Calibrate sensor based on reference curve.
   To obtain the reference curve, please have a look at the
   reproducibility folder.
   The intensity profiles are fit to logistic growth curves
   and the time constants of the accumulation and degradation
   are determined.

1. `explanation-dtw-alignment.ipynb`:
   Explains the principle behind the phase percentage estimate
   by subsequence alignment.
   
## Reproducibility

For documentation purposes, we share the notebooks
that have been used to produce the results in the
preprint.

1. `extract_calibration_data.ipynb`:
   Example of obtaining a reference curve from a TrackMate file.
   **Note**: We do not provide the TrackMate XML files here.
   Instead, a reference curve is provided,
   which is used in the other examples.
