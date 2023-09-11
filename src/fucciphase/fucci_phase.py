import pandas as pd


def compute_phase_color(df: pd.DataFrame, channel1: str, channel2: str) -> None:
    """Compute the phase color for each spot using the two channels respective
    intensity, and update the dataframe with a unified mean intensity value, phase,
    and RGB color columns.

    TODO: describe the LUT.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    channel1 : str
        Name of the first channel.
    channel2 : str
        Name of the second channel.

    Raises
    ------
    ValueError
        If the dataframe does not contain the normalized channels.
    """
    pass  # TODO
