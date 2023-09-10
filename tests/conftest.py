import pandas as pd
import pytest


@pytest.fixture
def trackmate_df():
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
        }
    )

    return df


@pytest.fixture
def trackmate_csv(tmp_path, trackmate_df: pd.DataFrame):
    """Save a mock trackmate csv file."""
    # export to csv
    csv_path = tmp_path / "trackmate.csv"
    trackmate_df.to_csv(csv_path, index=False)

    return csv_path
