import numpy as np

from fucciphase.utils.trackmate import TrackMateXML


def test_import_spotless_xml(spotless_trackmate_xml):
    """Test that a spotless xml file is correctly imported."""
    tm_xml = TrackMateXML(spotless_trackmate_xml)

    # check readout
    assert tm_xml._tree is not None
    assert tm_xml._root is not None
    assert tm_xml._model is not None
    assert tm_xml._allspots is not None
    assert tm_xml.nspots == 0
    assert len(tm_xml.features) == 0


def test_import_xml(trackmate_xml):
    """Test that an xml file with spots is correctly imported."""
    tm_xml = TrackMateXML(trackmate_xml)

    # check readout
    assert tm_xml._tree is not None
    assert tm_xml._root is not None
    assert tm_xml._model is not None
    assert tm_xml._allspots is not None
    assert tm_xml.nspots == 4
    assert len(tm_xml.features) == 35


def test_import_as_pandas_spotless(spotless_trackmate_xml):
    """Test that a spotless xml file lead to empty dataframe"""
    # import xml
    tm_xml = TrackMateXML(spotless_trackmate_xml)

    # export dataframe
    df = tm_xml.to_pandas()
    assert len(df) == 0


def test_import_as_pandas(trackmate_xml):
    """Test that the spots in the xml file are imported as a dataframe correctly."""
    # import xml
    tm_xml = TrackMateXML(trackmate_xml)

    # export dataframe
    df = tm_xml.to_pandas()
    assert len(df) == tm_xml.nspots == 4


def test_save_xml(tmp_path, trackmate_xml):
    """Test that an imported xml can be saved."""
    # import xml
    tm_xml = TrackMateXML(trackmate_xml)

    # save xml
    save_path = tmp_path / "test.xml"
    tm_xml.save_xml(save_path)

    # re-import xml
    tm_xml_2 = TrackMateXML(save_path)

    # compare the dataframes
    df_1 = tm_xml.to_pandas()
    df_2 = tm_xml_2.to_pandas()
    assert df_1.equals(df_2)


def test_update_features(tmp_path, trackmate_xml):
    """Test updating features in the xml tree."""
    # load xml
    tm_xml = TrackMateXML(trackmate_xml)

    # export dataframe to pandas
    df = tm_xml.to_pandas()

    # add new columns
    new_feature1 = "new_feature1"
    df[new_feature1] = 5 * np.arange(len(df))
    new_feature2 = "new_feature2"
    df[new_feature2] = 1 + 10 * np.arange(len(df))

    # update xml
    tm_xml.update_features(df)

    # re-export dataframe to pandas
    df_2 = tm_xml.to_pandas()
    assert df.equals(df_2[df.columns])  # make sure the columns are ordered similarly

    # write xml
    save_path = tmp_path / "test.xml"
    tm_xml.save_xml(save_path)

    # re-import xml
    tm_xml_2 = TrackMateXML(save_path)
    df_3 = tm_xml_2.to_pandas()

    # the new columns are present but of different dtypes
    assert new_feature1 in df_3.columns
    assert new_feature2 in df_3.columns

    for c in [new_feature1, new_feature2]:
        assert (df_3[c].astype(int) == df[c]).all()
