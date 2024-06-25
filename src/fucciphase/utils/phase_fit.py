import numpy as np
import pandas as pd
from monotonic_derivative import ensure_monotonic_derivative


def fit_percentages(frames: np.ndarray, percentages: np.ndarray) -> np.ndarray:
    """Fit estimated percentages to function with non-negative derivative."""
    best_fit: np.ndarray = ensure_monotonic_derivative(
        x=frames,
        y=percentages,
        degree=1,
        force_negative_derivative=False,
    )
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
