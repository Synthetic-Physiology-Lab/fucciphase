import re
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# TrackMate XML tags
MODEL = "Model"
FEATURES = "FeatureDeclarations"
SPOT_FEATURES = "SpotFeatures"
ALL_SPOTS = "AllSpots"
N_SPOTS = "nspots"
ID = "ID"
SUBTRACK_REGEX = r"Track_[0-9]+\.[a-z]+"


# TODO test on very large trackmate files, since this is potentially a bottleneck here
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
    features : Dict[str, type]
        List of all features in the xml file, and whether they are integer features.
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
        self._tree: et.ElementTree[et.Element[str]] = et.parse(xml_path)
        self._root: et.Element | Any = self._tree.getroot()

        # placeholders
        self._model: Optional[et.Element] = None
        self._allspots: Optional[et.Element] = None

        self.nspots: int = 0  # number of spots
        self.features: Dict[str, type] = {}  # features and their types
        self.spot_features: List[str] = []  # list of spot features

        # import model and all spots
        self._import_data()

    def _get_spot_features(self) -> None:
        """Get the spot features from the tree."""
        if self._allspots is not None:
            spot_features: List[str] = []
            for frame in self._allspots:
                for spot in frame:
                    spot_features.extend(spot.attrib.keys())
                    break
                break

            self.spot_features = spot_features

    def _get_features(self) -> None:
        """Compare spot features and features declaration, keep the intersection and
        register the dtypes in a public member.
        """
        if self._model is not None:
            features = {}
            for element in self._model:
                if element.tag == FEATURES:
                    for feature in element:
                        if feature.tag == SPOT_FEATURES:
                            for spot_feature in feature:
                                # get feature name
                                feature_name = spot_feature.attrib["feature"]

                                # check if feature is integer
                                is_integer = spot_feature.attrib["isint"] == "true"

                                # add feature to dictionary
                                features[feature_name] = int if is_integer else float

            # get spot features
            self._get_spot_features()

            # keep only features that are in both spot features and features
            features_key = set(features.keys())
            features_to_keep = features_key - (features_key - set(self.spot_features))
            self.features = {feature: features[feature] for feature in features_to_keep}

    def _import_data(self) -> None:
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
                self._model = element

        if self._model is None:
            raise ValueError('"Model" tag not found in xml file.')

        # get allspots
        for element in self._model:
            if element.tag == ALL_SPOTS:
                self._allspots = element
                self.nspots = int(element.attrib[N_SPOTS])

        if self._allspots is None:
            raise ValueError('"AllSpots" tag not found in xml file.')

        # get feature declarations
        self._get_features()

    def _add_track_ids(self, df: pd.DataFrame) -> None:
        # extract track IDs (if there are spots)
        if len(df) > 0:
            # some spots do not have tracks associated
            track_ids = np.full_like(df[ID].values, -1)
            if self._model is not None:
                for element in self._model:
                    if element.tag == "AllTracks":
                        for track in element:
                            track_id = int(track.attrib["TRACK_ID"])

                            for edge in track:
                                spot_source = edge.attrib["SPOT_SOURCE_ID"]
                                spot_target = edge.attrib["SPOT_TARGET_ID"]

                                # get row index of the source and target spots
                                source_index = df[df[ID] == spot_source].index[0]
                                target_index = df[df[ID] == spot_target].index[0]

                                # update track IDs
                                track_ids[source_index] = track_id
                                track_ids[target_index] = track_id

                # update dataframe
                df["TRACK_ID"] = track_ids
        else:
            df["TRACK_ID"] = []

        # add track id to the spot features (this avoids exporting it to the xml later)
        self.spot_features.append("TRACK_ID")

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

        # add tracks IDs
        self._add_track_ids(df)

        # fix type of TRACK_ID
        self.features["TRACK_ID"] = int
        # convert features to their declared types
        return df.astype(self.features)

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
        new_features = set(df.columns) - set(self.spot_features)

        if self._allspots is not None:
            # update features
            for feature in new_features:
                self.features[feature] = df[feature].dtype

            # if there are spots and features
            if len(df) > 0 and len(new_features) > 0:
                # loop over the frames and spots
                for frame in self._allspots:
                    for spot in frame:
                        # get ID
                        spot_id = spot.attrib[ID]

                        # get spot
                        spot_df = df[df[ID] == spot_id]

                        # add the new feature
                        for feature in new_features:
                            spot.attrib[feature] = str(spot_df[feature].values[0])

    def save_xml(self, xml_path: Union[str, Path]) -> None:
        """Save the xml file.

        Parameters
        ----------
        xml_path : Union[str, Path]
            Path to the xml file.
        """
        with open(xml_path, "wb") as f:
            self._tree.write(f)

    def get_full_tracks(
        df: pd.DataFrame,
        channels: List[str],
        track_id_name: str = "UNIQUE_TRACK_ID",
        spot_name: str = "name",
        frame_name: str = "FRAME",
        min_length: int = 40,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Locate all tracks that may describe a full cycle.
        Tracks need to be auto-named by TrackMate.
        For example, Track_1a, Track_1aa, Track_1b etc.
        This can be done in TrackMate by executing an action after the tracking.
        In addition, tracks longer than a certain minimum length can be selected.
        """
        regex = "Track_[0-9]+.[a-z]+"
        candidate_tracks: List[pd.DataFrame] = []
        save_tracks: List[pd.DataFrame] = []
        track_ids = df[track_id_name].unique()
        for track_id in track_ids:
            track = df[df[track_id_name] == track_id]
            name = track[spot_name].iloc[0]
            last_frame = track[frame_name].max()
            # is the track a subtrack
            match = re.match(regex, name)
            # is there a subtrack
            next_match = any(df[spot_name].str.match(name + "[a-z]+").unique())
            if match is not None and last_frame < df[frame_name].max():
                if next_match:
                    save_tracks.append(track[[frame_name, *channels]])
                else:
                    if len(track) > min_length:
                        candidate_tracks.append(track[[frame_name, *channels]])
        return save_tracks, candidate_tracks
