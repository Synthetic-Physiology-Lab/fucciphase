import pandas as pd
import pytest


@pytest.fixture
def trackmate_example():
    """Create a mock trackmate dataframe.

    This dataframe simulates the naive import of a trackmate csv file."""
    # create mock dataframe
    df = pd.DataFrame(
        {
            "LABEL": ["Label", "Label", "", "ID4072", "ID2322", "ID4144"],
            "ID": ["Spot ID", "Spot ID", "", 4072, 2322, 4144],
            "TRACK_ID": ["Track ID", "Track ID", "", 85, 85, 85],
            "POSITION_X": ["X", "X", "(micron)", 76.15995973, 57.23480319, 23.91266266],
            "POSITION_Y": ["Y", "Y", "(micron)", 94.66421077, 81.67105699, 94.66421077],
            "POSITION_T": ["T", "T", "(sec)", 2699.980774, 26099.81415, 67499.51935],
            "FRAME": ["Frame", "Frame", "", 3, 29, 75],
            "MEAN_INTENSITY_CH1": [
                "Mean intensity ch1",
                "Mean ch1",
                "(counts)",
                544.5488114,
                1194.44928,
                419.8042824,
            ],
            "MEAN_INTENSITY_CH2": [
                "Mean intensity ch2",
                "Mean ch2",
                "(counts)",
                137.4776146,
                147.8754012,
                206.0246564,
            ],
        }
    )

    return df


@pytest.fixture
def trackmate_df(trackmate_example: pd.DataFrame) -> pd.DataFrame:
    """Create a mock trackmate dataframe without the extraneous
    rows (header duplicates and units).

    Returns
    -------
    pd.DataFrame
        Mock trackmate dataframe.
    """
    # remove the first three rows, re-index the dataframe and
    # convert the types
    trackmate_example.drop(index=[0, 1, 2], inplace=True)
    trackmate_example.reset_index(drop=True, inplace=True)
    trackmate_example = trackmate_example.convert_dtypes()

    return trackmate_example


@pytest.fixture
def trackmate_csv(tmp_path, trackmate_example: pd.DataFrame):
    """Save a mock trackmate csv file."""
    # export to csv
    csv_path = tmp_path / "trackmate.csv"
    trackmate_example.to_csv(csv_path, index=False)

    return csv_path
