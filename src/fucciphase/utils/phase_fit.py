import lmfit
import numpy as np
import pandas as pd


def fit_percentages(frames: np.ndarray, percentages: np.ndarray) -> np.ndarray:
    """Fit estimated percentages to straight line."""
    model = lmfit.models.LinearModel()
    parameters = model.guess(frames, percentages)
    parameters["slope"].min = 0
    parameters["intercept"].min = 0
    try:
        fit = model.fit(
            percentages,
            parameters,
            x=frames,
            method="least_squares",
            fit_kws={"loss": "soft_l1"},
        )
        best_fit: np.ndarray = fit.best_fit
        slope = fit.params["slope"]
        if np.isclose(slope, 0):
            best_fit = np.full(percentages.shape, np.nan)
    except ValueError:
        best_fit = np.full(percentages.shape, np.nan)
    # percentage cannot be larger than 100%
    if np.max(best_fit) > 100.01:
        best_fit[:] = np.nan
    return best_fit


def postprocess_estimated_percentages(df: pd.DataFrame) -> None:
    """Make estimated percentages continuous."""
    indices = df["TRACK_ID"].unique()
    df["CELL_CYCLE_PERC_POST"] = np.nan
    for index in indices:
        if index == -1:
            continue
        track = df[df["TRACK_ID"] == index]
        frames = track["FRAME"]
        percentages = track["CELL_CYCLE_PERC"]
        restored_percentages = fit_percentages(frames, percentages)
        df.loc[df["TRACK_ID"] == index, "CELL_CYCLE_PERC_POST"] = restored_percentages
