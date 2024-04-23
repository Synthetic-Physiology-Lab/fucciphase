import lmfit
import numpy as np


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
    if np.max(best_fit) > 100.1:
        best_fit[:] = np.nan
    return best_fit
