import numpy as np
import pandas as pd

from fucciphase.sensor import FUCCISASensor

# TODO improve simulation, use logistic functions


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

    return np.round(ch, 2)


def simulate_single_track(track_id: float = 42, mean: float = 0.5) -> pd.DataFrame:
    """Simulate a single track.

    Parameters
    ----------
    track_id : int
        Track ID
    mean : float
        Temporal mean corresponding to the crossing between the two channels

    Returns
    -------
    pd.DataFrame
        Dataframe mocking a Trackmate single track import.
    """
    # example data
    phase_percentages = [33.3, 33.3, 33.3]
    center = [20.0, 55.0, 70.0, 95.0]
    sigma = [5.0, 5.0, 10.0, 1.0]
    # create sensor
    sensor = FUCCISASensor(
        phase_percentages=phase_percentages,
        center=center,
        sigma=sigma,
    )
    # create the time vector
    percentage = np.arange(0, 50) / 50
    t = 24 * percentage
    percentage *= 100

    # create the channels as Gaussian of time
    ch1, ch2 = sensor.get_expected_intensities(percentage)

    # create dataframe
    df = pd.DataFrame(
        {
            "LABEL": [f"ID{i}" for i in range(len(t))],
            "ID": list(range(len(t))),
            "TRACK_ID": [track_id for _ in range(len(t))],
            "POSITION_X": [np.round(mean * i * 0.02, 2) for i in range(len(t))],
            "POSITION_Y": [np.round(mean * i * 0.3, 2) for i in range(len(t))],
            "POSITION_T": [np.round(i * 0.01, 2) for i in range(len(t))],
            "FRAME": list(range(len(t))),
            "PERCENTAGE": percentage,
            "MEAN_INTENSITY_CH1": [0 for _ in range(len(t))],
            "MEAN_INTENSITY_CH2": [1 for _ in range(len(t))],
            "MEAN_INTENSITY_CH3": ch1,
            "MEAN_INTENSITY_CH4": ch2,
        }
    )
    return df
