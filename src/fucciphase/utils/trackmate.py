import xml.etree.ElementTree as et
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# TrackMate XML tags
MODEL = "Model"
ALL_SPOTS = "AllSpots"
N_SPOTS = "nspots"
ID = "ID"


class TrackMateXML:
    """Class to handle TrackMate xml files.

    TrackMate xml files are structured as follows:
        root
        ├── Log
        ├── Model
        │   ├── FeatureDeclarations
        │   ├── AllSpots
        │   │   └── SpotsInFrame
        │   │       └── Spot
        │   ├── AllTracks
        │   └── FilteredTracks
        ├── Settings
        ├── GUIState
        └── DisplaySettings

    This class allows reading in the tree and converting the spots to a
    pandas dataframe. The features (columns) can also be updated and the
    xml file be saved.

    Attributes
    ----------
    nspots : int
        Number of spots in the xml file.
    features : List[str]
        List of all features in the xml file. Empty if there is not spot in the xml.
    """

    def __init__(self, xml_path: Union[str, Path]) -> None:
        """Initialize the TrackMateXML object.

        The xml file is parsed and the model and all spots are imported.

        Parameters
        ----------
        xml_path : Union[str, Path]
            Path to the xml file.
        """
        # parse tree
        self._tree: et.ElementTree = et.parse(xml_path)
        self._root: et.Element = self._tree.getroot()

        # placeholders
        self._model: Optional[et.Element] = None
        self._allspots: Optional[et.Element] = None

        self.nspots: int = 0
        self.features: List[str] = []

        # import model and all spots
        self._import_model()

    def _import_model(self) -> None:
        """Import the model and all spots from the xml file.

        Raises
        ------
        ValueError
            If the xml file does not contain a "Model" tag.
        ValueError
            If the "Model" tag does not contain an "AllSpots" tag.
        """
        # get model
        for element in self._root:
            if element.tag == MODEL:
                self.model = element

        if self.model is None:
            raise ValueError('"Model" tag not found in xml file.')

        # get allspots
        for element in self.model:
            if element.tag == ALL_SPOTS:
                self._allspots = element
                self.nspots = int(element.attrib[N_SPOTS])

        if self._allspots is None:
            raise ValueError('"AllSpots" tag not found in xml file.')

        # get spot features
        for frame in self._allspots:
            for spot in frame:
                self.features = list(spot.attrib.keys())
                break
            break

    def to_pandas(self) -> pd.DataFrame:
        """Export the spots as a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the spots.
        """
        df = pd.DataFrame()

        # loop over all frames and add spots to a dataframe
        if self._allspots is not None:
            spot_count = 0
            for frame in self._allspots:
                # only run on frames with spots
                if len(frame) > 0:
                    # for each spot in the frame
                    for spot in frame:
                        # if this is the first spot, initialize dataframe
                        if spot_count == 0:
                            df = pd.DataFrame(columns=spot.attrib.keys())

                        # add the spot to the dataframe
                        df.loc[spot_count] = spot.attrib
                        spot_count += 1

        return df

    def update_features(self, df: pd.DataFrame) -> None:
        """Update the xml tree with new features, where features are columns of the
        dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the new features.
        """
        # compare number of spots
        if len(df) != self.nspots:
            raise ValueError(
                f"Number of spots in the dataframe ({len(df)}) does not match number "
                f"of spots in xml file ({self.nspots})."
            )

        # check if ID column is in the dataframe
        if ID not in df.columns:
            raise ValueError(f"Column {ID} not found in dataframe.")

        # new features
        new_features = set(df.columns) - set(self.features)

        if self._allspots is not None:
            # if there are spots and features
            if len(df) > 0 and len(new_features) > 0:
                # loop over the frames and spots
                for frame in self._allspots:
                    for spot in frame:
                        # get ID
                        spot_id = spot.attrib[ID]

                        # get spot
                        spot_df = df[df[ID] == spot_id]

                        # add the new features as strings
                        for feature in new_features:
                            spot.attrib[feature] = str(spot_df[feature])

    def save_xml(self, xml_path: Union[str, Path]) -> None:
        """Save the xml file.

        Parameters
        ----------
        xml_path : Union[str, Path]
            Path to the xml file.
        """
        with open(xml_path, "wb") as f:
            self._tree.write(f)
