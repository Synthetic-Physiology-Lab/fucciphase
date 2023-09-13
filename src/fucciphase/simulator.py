import numpy as np
import pandas as pd

# TODO improve simulation, use sine waves


def simulate_single_channel(
    t: np.ndarray, mean: float, sigma: float, amp: float = 1.0
) -> np.ndarray:
    """Simulate a single channel.

    Parameters
    ----------
    t : np.ndarray
        Time vector
    mean : float
        Mean of the Gaussian
    sigma : float
        Standard deviation of the Gaussian
    amp : float
        Amplitude of the Gaussian

    Returns
    -------
    np.ndarray
        Intensity vector
    """
    ch: np.ndarray = amp * np.exp(-((t - mean) ** 2) / (2 * sigma**2))

    return ch


def simulate_single_track() -> pd.DataFrame:
    """Simulate a single track.

    Returns
    -------
    pd.DataFrame
        Dataframe mocking a Trackmate single track import.
    """
    # create the time vector
    t = 24 * np.arange(0, 50) / 50
    sigma = 0.2 * 24
    mean1 = 0.5 * 24 - sigma
    mean2 = 0.5 * 24 + sigma
    amp1 = 1
    amp2 = 0.9

    # create the channels as Gaussian of time
    ch1 = simulate_single_channel(t, mean1, sigma, amp1)
    ch2 = simulate_single_channel(t, mean2, sigma, amp2)

    # create dataframe
    df = pd.DataFrame(
        {
            "LABEL": [f"ID{i}" for i in range(len(t))],
            "ID": [str(i) for i in range(len(t))],
            "TRACK_ID": [85 for _ in range(len(t))],
            "POSITION_X": [105.567 for _ in range(len(t))],
            "POSITION_Y": [82.23 for _ in range(len(t))],
            "FRAME": list(range(len(t))),
            "MEAN_INTENSITY_CH1": [0 for _ in range(len(t))],
            "MEAN_INTENSITY_CH2": [1 for _ in range(len(t))],
            "MEAN_INTENSITY_CH3": ch1,
            "MEAN_INTENSITY_CH4": ch2,
        }
    )
    return df
