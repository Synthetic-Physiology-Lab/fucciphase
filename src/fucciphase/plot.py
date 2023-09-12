import pandas as pd
from matplotlib import pyplot as plt

from .phase import NewColumns
from .utils import get_norm_channel_name


def plot_phase(
    df: pd.DataFrame, channel1: str, channel2: str, detect_phases: bool = False
) -> None:
    """Plot the two channels, the unique intensity and vertical lines
    corresponding to the change of phase.

    The dataframe must be preprocessed with one of the available phase
    computation function and must contain the following columns:
        - normalised channels (channel1 + "_NORM", etc)
        - unique intensity
        - phase
        - FRAME

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    channel1 : str
        First channel
    channel2 : str
        Second channel
    detect_phases : bool, optional
        If True, detect the phases and plot vertical lines, by default False

    Raises
    ------
    ValueError
        If the dataframe does not contain the mandatory columns.
    ValueError
        If the dataframe does not contain the mandatory columns.
    """
    # check if the FRAME column is present
    if "FRAME" not in df.columns:
        raise ValueError("Column FRAME not found")

    # check if all new columns are present
    for column in NewColumns:
        if column.value not in df.columns:
            raise ValueError(f"Column {column.value} not found")

    # get frame, normalised channels, unique intensity and phase
    t = df["FRAME"].to_numpy()
    channel1_norm = df[get_norm_channel_name(channel1)]
    channel2_norm = df[get_norm_channel_name(channel2)]
    unique_intensity = df[NewColumns.unique_intensity()]
    phase = df[NewColumns.phase()].to_numpy()

    # plot
    plt.plot(t, channel1_norm, label=channel1)
    plt.plot(t, channel2_norm, label=channel2)
    plt.plot(t, unique_intensity, label="unique intensity")

    if detect_phases:
        for i in range(len(phase)):
            if i > 0:
                if phase[i] != phase[i - 1]:
                    plt.axvline(x=t[i], color="k", linestyle="--")
