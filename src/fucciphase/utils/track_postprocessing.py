from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LineageTree import lineageTree
from matplotlib import colormaps
from scipy import signal


def split_track(
    track: pd.DataFrame,
    highest_track_idx: int,
    sg2m_channel: str,
    distance: int = 3,
    background_fluctuation_percentage: float = 0.2,
    track_id_name: str = "TRACK_ID",
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
    track_id_name: str
        Name of column with track IDs

    """
    if track_id_name not in track.columns:
        raise ValueError(f"{track_id_name} column is missing.")
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
        track.loc[track.index[peak:next_peak], track_id_name] = highest_track_idx + 1
        highest_track_idx += 1

    return highest_track_idx


def split_all_tracks(
    track_df: pd.DataFrame,
    sg2m_channel: str,
    distance: int = 3,
    minimum_track_length: int = 20,
    background_fluctuation_percentage: float = 0.2,
    track_id_name: str = "TRACK_ID",
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
    track_id_name: str
        Name of column with track IDs

    """
    if track_id_name not in track_df.columns:
        raise ValueError(f"{track_id_name} column is missing.")
    highest_track_idx = track_df[track_id_name].max()
    highest_track_idx_counter = highest_track_idx
    # go through all tracks and split if needed
    for track_idx in range(highest_track_idx):
        track = track_df.loc[track_df[track_id_name] == track_idx]
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
        track_df.loc[track_df[track_id_name] == track_idx] = track


def compute_motility_parameters(
    track_df: pd.DataFrame,
    centroid_x: str = "POSITION_X",
    centroid_y: str = "POSITION_Y",
    centroid_z: bool = False,
    track_id_name: str = "TRACK_ID",
) -> None:
    """Add motility parameters to DataFrame.

    Parameters
    ----------
    track_df: pd.DataFrame
        DataFrame with tracking data
    centroid_x: str
        Name of column with x-coordinate of centroid
    centroid_y: str
        Name of column with y-coordinate of centroid
    centroid_z: str
        Name of column with z-coordinate of centroid
    track_id_name: str
        Name of column with track IDs

    """
    track_df["MSD"] = np.nan
    track_df["DISPLACEMENTS"] = np.nan
    indices = track_df[track_id_name].unique()
    for index in indices:
        if index == -1:
            continue
        track = track_df[track_df[track_id_name] == index]
        centroids_x = track[centroid_x].to_numpy()
        centroids_y = track[centroid_y].to_numpy()
        centroids_z = None
        if centroid_z is not False:
            centroids_z = track[centroid_z].to_numpy()

        displacements = compute_displacements(centroids_x, centroids_y, centroids_z)
        velocities = compute_velocities(centroids_x, centroids_y, centroids_z)
        MSDs = compute_MSD(centroids_x, centroids_y, centroids_z)
        track_df.loc[track_df[track_id_name] == index, "DISPLACEMENTS"] = displacements
        track_df.loc[track_df[track_id_name] == index, "VELOCITIES"] = velocities
        track_df.loc[track_df[track_id_name] == index, "MSD"] = MSDs


def compute_displacements(
    centroids_x: np.ndarray, centroids_y: np.ndarray, centroids_z: Optional[np.ndarray]
) -> np.ndarray:
    """Compute displacement w.r.t origin."""
    N = len(centroids_x)
    x0 = centroids_x[0]
    y0 = centroids_y[0]
    z0 = None
    if centroids_z is not None:
        z0 = centroids_z[0]
    r0 = (x0, y0, z0)
    distances = np.zeros(N)
    for idx in range(N):
        x = centroids_x[idx]
        y = centroids_y[idx]
        z = None
        if centroids_z is not None:
            z = centroids_z[idx]
        r = (x, y, z)
        distances[idx] = np.sqrt(get_squared_displacement(r0, r))
    return distances


def compute_velocities(
    centroids_x: np.ndarray, centroids_y: np.ndarray, centroids_z: Optional[np.ndarray]
) -> np.ndarray:
    """Compute velocity."""
    N = len(centroids_x)
    x0 = centroids_x[0]
    y0 = centroids_y[0]
    z0 = None
    if centroids_z is not None:
        z0 = centroids_z[0]
    r0 = (x0, y0, z0)
    distances = np.zeros(N)
    for idx in range(N):
        x = centroids_x[idx]
        y = centroids_y[idx]
        z = None
        if centroids_z is not None:
            z = centroids_z[idx]
        r = (x, y, z)
        distances[idx] = np.sqrt(get_squared_displacement(r0, r))
        # overwrite start vector
        r0 = (x, y, z)
    return distances


def compute_MSD(
    centroids_x: np.ndarray, centroids_y: np.ndarray, centroids_z: Optional[np.ndarray]
) -> np.ndarray:
    """Compute mean-squared distance.

    Notes
    -----
    Please find more information in
    Methods for cell and particle tracking.,
    Meijering E, Dzyubachyk O, Smal I.,
    Methods Enzymol. 2012;504:183-200.
    https://doi.org/10.1016/B978-0-12-391857-4.00009-4
    """
    N = len(centroids_x)
    MSDs = np.zeros(N)
    for idx in range(N):
        if idx == 0:
            continue
        MSD = 0.0
        for i in range(N - idx):
            x = centroids_x[i + idx]
            y = centroids_y[i + idx]
            z = None
            if centroids_z is not None:
                z = centroids_z[i + idx]
            r = (x, y, z)

            xi = centroids_x[i]
            yi = centroids_y[i]
            zi = None
            if centroids_z is not None:
                zi = centroids_z[i]
            ri = (xi, yi, zi)
            MSD += get_squared_displacement(ri, r)
        MSD /= N - idx
        MSDs[idx] = MSD
    return MSDs


def get_squared_displacement(r0: tuple, r: tuple) -> float:
    """Return squared displacement between two points."""
    if not len(r0) == 3:
        raise ValueError("Provide three-component coordinates")
    if not len(r) == 3:
        raise ValueError("Provide three-component coordinates")
    displacement = 0.0
    for i in range(3):
        x0 = r0[i]
        x = r[i]
        if x0 is None:
            continue
        displacement += (x0 - x) ** 2
    return displacement


def plot_trackscheme(
    df: pd.DataFrame,
    track_id_name: str = "TRACK_ID",
    time_id: str = "POSITION_T",
    cycle_percentage_id: str = "CELL_CYCLE_PERC_POST",
    figsize: tuple = (10, 30),
) -> None:
    """Plot tracks similar to TrackMate trackscheme.

    Parameters
    ----------
    df: pd.DataFrame
       DataFrame holding tracks
    track_id_name: str
        Name of column with track IDs
    time_id : str
        Name of column with time steps
    cycle_percentage_id: str
        Name of column with cell cycle percentage info
    figsize: tuple
        Size of matplotlib figure

    Notes
    -----
    A percentage column, which must contain values between 0 and 100
    is used to color the individual dots.
    """
    cmap_name = "cool"
    cmap = colormaps.get(cmap_name)
    plt.figure(figsize=figsize)
    for track_id in df[track_id_name]:
        track = df.loc[df[track_id_name] == track_id, time_id]
        color = df.loc[df[track_id_name] == track_id, cycle_percentage_id]
        colormapper = []
        for c in color:
            if np.isnan(c):
                colormapper.append("black")
            else:
                colormapper.append(cmap(c / 100.0))
        sc = plt.scatter([round(track_id)] * len(track), track, color=colormapper)
    plt.xticks(np.arange(1, df[track_id_name].max(), step=1))
    sc.set_cmap(cmap_name)

    cbar = plt.colorbar(ticks=[0, 0.5, 1], location="top")
    cbar.ax.set_xticklabels([0, 50, 100])
    return


def split_trackmate_tracks(
    df: pd.DataFrame, track_id_name: str = "TRACK_ID", label_id_name: str = "name"
) -> None:
    """Split TrackMate tracks into subtracks.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame obtained from TrackMate XML, updated in place
    label_id_name: str
        Name of spots to split track IDs into unique tracks
    track_id_name: str
        Name of track ID column

    Notes
    -----
    TrackMate permits track splitting but then assigns
    the same Track ID for all subtracks.
    A way around this is to use a TrackMate action:
    https://forum.image.sc/t/how-to-identify-subtracks-in-tracking-csv/71474

    Use this action on your data first and then use this function
    to obtain a new DataFrame that has unique Track IDs.
    The updated Track IDs are stored in a new column called
    `UNIQUE_TRACK_ID`.
    """
    # pattern to identify subtracks
    regex = r"Track_[0-9]+\.[a-z]+"
    subtracks = df.loc[df[label_id_name].str.contains(regex), label_id_name].unique()
    subtracks = sorted(subtracks)

    mapping_of_subtracks = {}
    max_track = df[track_id_name].max() + 1

    for subtrack in subtracks:
        mapping_of_subtracks[subtrack] = max_track
        max_track += 1

    subtrack_series = df.loc[df[label_id_name].str.contains(regex), label_id_name]
    new_track_ids = subtrack_series.transform(lambda x: mapping_of_subtracks[x])

    df.loc[:, "UNIQUE_TRACK_ID"] = df[track_id_name].copy()
    df["UNIQUE_TRACK_ID"].update(new_track_ids)
    return


# flake8: noqa: C901
def export_lineage_tree_to_svg(
    df: pd.DataFrame,
    trackmate_file: str,
    node_color_column: Optional[str] = None,
    stroke_width: Optional[float] = None,
) -> List[str]:
    """Write a lineage tree colored by FUCCI phases.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame processed by fucciphase
    trackmate_file: str
        The original trackmate file
    node_color_column: Optional[str]
        Name of column to color nodes
    stroke_width: Optional[float]
        Width of edges connecting nodes


    Returns
    -------
    final_track_names: List[str]
        Names of tracks (from left to right)

    Notes
    -----
    This function currently only supports
    the standard FUCCISA sensor.
    """
    print("Warning: make sure that you updated the spot names using TrackMate actions!")
    # initialise lineage tree
    lt = lineageTree(trackmate_file, file_type="TrackMate")
    cmap_name = "cool"
    cmap = colormaps.get(cmap_name)

    # filter spots that are not part of a track
    for track in lt.all_tracks:
        spot_0 = track[0]
        track_name = df.loc[df["ID"].astype(int) == spot_0, "name"].values
        if not len(track_name) == 1:
            raise RuntimeError("Illegal track found")
        if "Track" not in track_name[0]:
            lt.remove_track(track)

    node_color = None
    if node_color_column is not None:
        if "PERC" in node_color_column:

            def node_color(id: int) -> tuple:
                color = df.loc[df["ID"].astype(int) == id, "CELL_CYCLE_PERC_DTW"].values
                if len(color) == 0:
                    raise ValueError("ID not in track")
                if np.isnan(color[0]):
                    rgba_value = (0, 0, 0)
                else:
                    rgba_value = cmap(color[0] / 100.0)
                return (255 * rgba_value[0], 255 * rgba_value[1], 255 * rgba_value[2])

        elif "PHASE" in node_color_column:

            def node_color(id: int) -> tuple:
                color = df.loc[df["ID"].astype(int) == id, "DISCRETE_PHASE_MAX"].values
                if len(color) == 0:
                    raise ValueError("ID not in track")
                color = color[0]
                if color == "G1":
                    color = 0.0
                elif color == "G1/S":
                    color = 0.4
                else:
                    color = 1.0
                rgba_value = cmap(color)
                return (255 * rgba_value[0], 255 * rgba_value[1], 255 * rgba_value[2])

        else:
            raise ValueError(
                "So far only discrete phases or percentages "
                "for 2-channel FUCCI sensors are supported."
            )

    stroke_width_function = None
    if stroke_width is not None:

        def stroke_width_function(id: int) -> float:
            return stroke_width

    lt.write_to_svg(
        "lineage_tree.svg",
        node_color=node_color,
        node_color_map=cmap_name,
        stroke_width=stroke_width_function,
    )
    final_track_names = []
    for root in lt.roots:
        track_name = df.loc[df["ID"].astype(int) == root, "name"].values
        final_track_names.append(track_name[0])
    return final_track_names
