import numpy as np
import pandas as pd
from scipy import signal


def split_track(
    track: pd.DataFrame,
    highest_track_idx: int,
    sg2m_channel: str,
    distance: int = 3,
    background_fluctuation_percentage: float = 0.2,
) -> int:
    """Detect mitosis events and split a single track.

    Parameters
    ----------
    track: pd.DataFrame
        DataFrame containing a single track
    highest_track_idx: int
        Highest index of all tracks! Split tracks will be appended
    sg2m_channel: str
        Name of the S/G2/M marker
    distance: int
        Minimum distance between peaks
    background_fluctuation_percentage: float
        Fluctuation of background level, used to detect low magenta level

    """
    if "TRACK_ID" not in track.columns:
        raise ValueError("TRACK_ID column is missing.")
    magenta = track[sg2m_channel]
    # get minima of magenta
    peaks, _ = signal.find_peaks(1.0 / magenta, distance=distance)
    magenta_background = magenta.min()
    # filter peaks
    peaks_to_use = []
    for idx, peak in enumerate(peaks):
        # if magenta intensity is high, continue
        if magenta.iloc[peak] > 1.2 * magenta_background:
            continue
        # check if there was a magenta signal in the meantime
        bg_level = 1 + background_fluctuation_percentage
        if not np.any(
            magenta.iloc[peaks[idx - 1] : peak] > bg_level * magenta_background
        ):
            continue
        peaks_to_use.append(peak)

    # split tracks
    for idx, peak in enumerate(peaks_to_use):
        next_peak = len(track)
        if len(peaks_to_use) > idx + 1:
            next_peak = peaks_to_use[idx + 1]
        track.loc[track.index[peak:next_peak], "TRACK_ID"] = highest_track_idx + 1
        highest_track_idx += 1

    return highest_track_idx


def split_all_tracks(
    track_df: pd.DataFrame,
    sg2m_channel: str,
    distance: int = 3,
    minimum_track_length: int = 20,
    background_fluctuation_percentage: float = 0.2,
) -> None:
    """Go through all tracks and split them after mitosis.

    Parameters
    ----------
    track_df: pd.DataFrame
        DataFrame containing multiple tracks, is changed in place
    sg2m_channel: str
        Name of the S/G2/M marker
    distance: int
        Minimum distance between peaks
    minimum_track_length: int
        minimum length required to check if track should be split
    background_fluctuation_percentage: float
        Fluctuation of background level, used to detect low magenta level

    """
    if "TRACK_ID" not in track_df.columns:
        raise ValueError("TRACK_ID column is missing.")
    highest_track_idx = track_df["TRACK_ID"].max()
    highest_track_idx_counter = highest_track_idx
    # go through all tracks and split if needed
    for track_idx in range(highest_track_idx):
        track = track_df.loc[track_df["TRACK_ID"] == track_idx]
        if len(track) < minimum_track_length:
            continue
        # split single track
        highest_track_idx_counter = split_track(
            track,
            highest_track_idx_counter,
            sg2m_channel,
            distance,
            background_fluctuation_percentage,
        )
        # update all tracks
        track_df.loc[track_df["TRACK_ID"] == track_idx] = track
