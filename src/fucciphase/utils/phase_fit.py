import logging

import numpy as np
import pandas as pd
from monotonic_derivative import ensure_monotonic_derivative

logger = logging.getLogger(__name__)


def fit_percentages(frames: np.ndarray, percentages: np.ndarray) -> np.ndarray:
    """Fit estimated percentages to function with non-negative derivative."""
    best_fit: np.ndarray = ensure_monotonic_derivative(
        x=frames,
        y=percentages,
        degree=1,
        force_negative_derivative=False,
    )
    # clip to range (0, 100)
    return np.clip(best_fit, 0.0, 100.0)  # type: ignore[no-any-return]


def postprocess_estimated_percentages(
    df: pd.DataFrame, percentage_column: str, track_id_name: str = "TRACK_ID"
) -> None:
    """Make estimated percentages continuous."""
    if percentage_column not in df:
        raise ValueError("The name of the percentage column is not in the DataFrame")
    indices = df[track_id_name].unique()
    postprocessed_percentage_column = percentage_column + "_POST"
    df[postprocessed_percentage_column] = np.nan
    for index in indices:
        if index == -1:
            continue
        track = df[df[track_id_name] == index]
        frames = track["FRAME"]
        percentages = track[percentage_column]
        if np.all(np.isnan(percentages)):
            logger.warning("No percentages to postprocess")
            return
        try:
            restored_percentages = fit_percentages(frames, percentages)
        except ValueError:
            logger.error(
                "Error in track %s. Make sure that the spots belong to a unique track, "
                "i.e., not more than one spot per frame per track.\n%s",
                index,
                track,
            )
        df.loc[df[track_id_name] == index, postprocessed_percentage_column] = (
            restored_percentages
        )
