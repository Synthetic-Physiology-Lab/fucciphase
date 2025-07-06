from pathlib import Path

import pandas as pd
import pytest

from fucciphase.utils import simulate_single_track


@pytest.fixture
def trackmate_example() -> pd.DataFrame:
    """Create a mock trackmate dataframe.

    This dataframe simulates the naive import of a trackmate csv file, which
    includes duplicated headers and units."""
    # simulate two tracks
    track1 = simulate_single_track(track_id=42, mean=0.4)
    track2 = simulate_single_track(track_id=402, mean=0.3)

    # concatenate the two dataframes
    df = pd.concat([track1, track2])

    # sort by frame
    df.sort_values(by=["FRAME"], inplace=True)

    # create dataframe with headers (from Trackmate csv export)
    df_header = pd.DataFrame(
        {
            "LABEL": ["Label", "Label", ""],
            "ID": ["Spot ID", "Spot ID", ""],
            "TRACK_ID": ["Track ID", "Track ID", ""],
            "POSITION_X": ["X", "X", "(micron)"],
            "POSITION_Y": ["Y", "Y", "(micron)"],
            "POSITION_T": ["T", "T", "(sec)"],
            "FRAME": ["Frame", "Frame", ""],
            "MEAN_INTENSITY_CH1": ["Mean intensity ch1", "Mean ch1", "(counts)"],
            "MEAN_INTENSITY_CH2": ["Mean intensity ch2", "Mean ch2", "(counts)"],
        }
    )

    # concatenate the two dataframes (again)
    df = pd.concat([df_header, df])

    # reset index
    df.reset_index(drop=True, inplace=True)

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
def trackmate_csv(tmp_path, trackmate_example: pd.DataFrame) -> Path:
    """Save a mock trackmate csv file."""
    # export to csv
    csv_path = tmp_path / "trackmate.csv"
    trackmate_example.to_csv(csv_path, index=False)

    return csv_path


def _write_to_file(path: Path, xml: str) -> None:
    """Write XML string to file, ensuring utf-8 encoding.

    Parameters
    ----------
    path : Path
        File in which to save the XML string
    xml : str
        XML string
    """
    # write file
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)

    return path


@pytest.fixture
def spotless_trackmate_xml(tmp_path) -> Path:
    """Create on the disk a TrackMate XML containing no spots."""
    # destination path
    path = tmp_path / "Spotless.xml"

    # file content corresponding to an empty Trackmate XML file
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <TrackMate version="7.11.1">
    <Log>TrackMate v7.11.1 started on:
    Tue, 12 Sep 2023 10:33:58
    Please note that TrackMate is available through Fiji, and is based on a publication. If you use it successfully for your research please be so kind to cite our work:
    Ershov, D., Phan, MS., Pylvänäinen, J.W., Rigaud S.U., et al. TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines. Nat Methods (2022). https://doi.org/10.1038/s41592-022-01507-1
    https://doi.org/10.1038/s41592-022-01507-1
    and / or:
    Tinevez, JY.; Perry, N. &amp; Schindelin, J. et al. (2017), 'TrackMate: An open and extensible platform for single-particle tracking.', Methods 115: 80-90, PMID 27713081.
    https://www.sciencedirect.com/science/article/pii/S1046202316303346

    Numerical feature analyzers:
    Spot feature analyzers:
    - Manual spot color provides: Spot color; is manual.
    - Spot intensity provides: Mean ch1, Median ch1, Min ch1, Max ch1, Sum ch1, Std ch1.
    - ExTrack probabilities provides: P stuck, P diffusive; is manual.
    - Spot contrast and SNR provides: Ctrst ch1, SNR ch1.
    Edge feature analyzers:
    - Directional change provides: y rate.
    - Edge speed provides: Speed, Disp.
    - Edge target provides: Source ID, Target ID, Cost.
    - Edge location provides: Edge T, Edge X, Edge Y, Edge Z.
    - Manual edge color provides: Edge color; is manual.
    Track feature analyzers:
    - Branching analyzer provides: N spots, N gaps, N splits, N merges, N complex, Lgst gap.
    - Track duration provides: Duration, Track start, Track stop, Track disp.
    - Track index provides: Index, ID.
    - Track location provides: Track X, Track Y, Track Z.
    - Track speed provides: Mean sp., Max speed, Min speed, Med. speed, Std speed.
    - Track quality provides: Mean Q.
    - Track motility analysis provides: Total dist., Max dist., Cfn. ratio, Mn. v. line, Fwd. progr., Mn. y rate.

    Image region of interest:
    Image data:
    For the image named: Example.
    Matching file Example in current folder.
    Geometry:
    X =    0 -  511, dx = 1.00000
    Y =    0 -  511, dy = 1.00000
    Z =    0 -    0, dz = 1.00000
    T =    0 -    1, dt = 1.00000

    Configured detector LoG detector with settings:
    - target channel: 1
    - threshold: 0.8
    - do median filtering: false
    - radius: 5.0
    - do subpixel localization: true

    Starting detection process using 16 threads.
    Detection processes 2 frames simultaneously and allocates 8 threads per frame.
    Found 0 spots.
    Detection done in 0.2 s.

    Computing spot quality histogram...
    Initial thresholding with a quality threshold above 0.2 ...
    Starting initial filtering process.
    Retained 0 spots out of 0.

    Calculating spot features...
    Calculating features done in 0.0 s.

    Performing spot filtering on the following features:
    No feature threshold set, kept the 0 spots.

    Configured tracker Simple LAP tracker with settings:
    - max frame gap: 2
    - alternative linking cost factor: 1.05
    - linking feature penalties:
    - linking max distance: 15.0
    - gap closing max distance: 15.0
    - merging feature penalties:
    - splitting max distance: 15.0
    - blocking value: Infinity
    - allow gap closing: true
    - allow track splitting: false
    - allow track merging: false
    - merging max distance: 15.0
    - splitting feature penalties:
    - cutoff percentile: 0.9
    - gap closing feature penalties:

    Starting tracking process.
    Tracking done in 0.0 s.
    Found 0 tracks.
    - avg size: 0.0 spots.
    - min size: 2147483647 spots.
    - max size: -2147483648 spots.

    Calculating features done in 0.0 s.
    Saving data...
    Warning: The source image does not match a file on the system.TrackMate won't be able to reload it when opening this XML file.
    To fix this, save the source image to a TIF file before saving the TrackMate session.</Log>
    <Model spatialunits="pixel" timeunits="frame">
        <FeatureDeclarations>
        <SpotFeatures>
            <Feature feature="QUALITY" name="Quality" shortname="Quality" dimension="QUALITY" isint="false" />
            <Feature feature="POSITION_X" name="X" shortname="X" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_Y" name="Y" shortname="Y" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_Z" name="Z" shortname="Z" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_T" name="T" shortname="T" dimension="TIME" isint="false" />
            <Feature feature="FRAME" name="Frame" shortname="Frame" dimension="NONE" isint="true" />
            <Feature feature="RADIUS" name="Radius" shortname="R" dimension="LENGTH" isint="false" />
            <Feature feature="VISIBILITY" name="Visibility" shortname="Visibility" dimension="NONE" isint="true" />
            <Feature feature="MANUAL_SPOT_COLOR" name="Manual spot color" shortname="Spot color" dimension="NONE" isint="true" />
            <Feature feature="MEAN_INTENSITY_CH1" name="Mean intensity ch1" shortname="Mean ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MEDIAN_INTENSITY_CH1" name="Median intensity ch1" shortname="Median ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MIN_INTENSITY_CH1" name="Min intensity ch1" shortname="Min ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MAX_INTENSITY_CH1" name="Max intensity ch1" shortname="Max ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="TOTAL_INTENSITY_CH1" name="Sum intensity ch1" shortname="Sum ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="STD_INTENSITY_CH1" name="Std intensity ch1" shortname="Std ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="EXTRACK_P_STUCK" name="Probability stuck" shortname="P stuck" dimension="NONE" isint="false" />
            <Feature feature="EXTRACK_P_DIFFUSIVE" name="Probability diffusive" shortname="P diffusive" dimension="NONE" isint="false" />
            <Feature feature="CONTRAST_CH1" name="Contrast ch1" shortname="Ctrst ch1" dimension="NONE" isint="false" />
            <Feature feature="SNR_CH1" name="Signal/Noise ratio ch1" shortname="SNR ch1" dimension="NONE" isint="false" />
        </SpotFeatures>
        <EdgeFeatures>
            <Feature feature="SPOT_SOURCE_ID" name="Source spot ID" shortname="Source ID" dimension="NONE" isint="true" />
            <Feature feature="SPOT_TARGET_ID" name="Target spot ID" shortname="Target ID" dimension="NONE" isint="true" />
            <Feature feature="LINK_COST" name="Edge cost" shortname="Cost" dimension="COST" isint="false" />
            <Feature feature="DIRECTIONAL_CHANGE_RATE" name="Directional change rate" shortname="y rate" dimension="ANGLE_RATE" isint="false" />
            <Feature feature="SPEED" name="Speed" shortname="Speed" dimension="VELOCITY" isint="false" />
            <Feature feature="DISPLACEMENT" name="Displacement" shortname="Disp." dimension="LENGTH" isint="false" />
            <Feature feature="EDGE_TIME" name="Edge time" shortname="Edge T" dimension="TIME" isint="false" />
            <Feature feature="EDGE_X_LOCATION" name="Edge X" shortname="Edge X" dimension="POSITION" isint="false" />
            <Feature feature="EDGE_Y_LOCATION" name="Edge Y" shortname="Edge Y" dimension="POSITION" isint="false" />
            <Feature feature="EDGE_Z_LOCATION" name="Edge Z" shortname="Edge Z" dimension="POSITION" isint="false" />
            <Feature feature="MANUAL_EDGE_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />
        </EdgeFeatures>
        <TrackFeatures>
            <Feature feature="TRACK_INDEX" name="Track index" shortname="Index" dimension="NONE" isint="true" />
            <Feature feature="TRACK_ID" name="Track ID" shortname="ID" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_SPOTS" name="Number of spots in track" shortname="N spots" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_GAPS" name="Number of gaps" shortname="N gaps" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_SPLITS" name="Number of split events" shortname="N splits" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_MERGES" name="Number of merge events" shortname="N merges" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_COMPLEX" name="Number of complex points" shortname="N complex" dimension="NONE" isint="true" />
            <Feature feature="LONGEST_GAP" name="Longest gap" shortname="Lgst gap" dimension="NONE" isint="true" />
            <Feature feature="TRACK_DURATION" name="Track duration" shortname="Duration" dimension="TIME" isint="false" />
            <Feature feature="TRACK_START" name="Track start" shortname="Track start" dimension="TIME" isint="false" />
            <Feature feature="TRACK_STOP" name="Track stop" shortname="Track stop" dimension="TIME" isint="false" />
            <Feature feature="TRACK_DISPLACEMENT" name="Track displacement" shortname="Track disp." dimension="LENGTH" isint="false" />
            <Feature feature="TRACK_X_LOCATION" name="Track mean X" shortname="Track X" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_Y_LOCATION" name="Track mean Y" shortname="Track Y" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_Z_LOCATION" name="Track mean Z" shortname="Track Z" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_MEAN_SPEED" name="Track mean speed" shortname="Mean sp." dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MAX_SPEED" name="Track max speed" shortname="Max speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MIN_SPEED" name="Track min speed" shortname="Min speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MEDIAN_SPEED" name="Track median speed" shortname="Med. speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_STD_SPEED" name="Track std speed" shortname="Std speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MEAN_QUALITY" name="Track mean quality" shortname="Mean Q" dimension="QUALITY" isint="false" />
            <Feature feature="TOTAL_DISTANCE_TRAVELED" name="Total distance traveled" shortname="Total dist." dimension="LENGTH" isint="false" />
            <Feature feature="MAX_DISTANCE_TRAVELED" name="Max distance traveled" shortname="Max dist." dimension="LENGTH" isint="false" />
            <Feature feature="CONFINEMENT_RATIO" name="Confinement ratio" shortname="Cfn. ratio" dimension="NONE" isint="false" />
            <Feature feature="MEAN_STRAIGHT_LINE_SPEED" name="Mean straight line speed" shortname="Mn. v. line" dimension="VELOCITY" isint="false" />
            <Feature feature="LINEARITY_OF_FORWARD_PROGRESSION" name="Linearity of forward progression" shortname="Fwd. progr." dimension="NONE" isint="false" />
            <Feature feature="MEAN_DIRECTIONAL_CHANGE_RATE" name="Mean directional change rate" shortname="Mn. y rate" dimension="ANGLE_RATE" isint="false" />
        </TrackFeatures>
        </FeatureDeclarations>
        <AllSpots nspots="0">
        <SpotsInFrame frame="0" />
        <SpotsInFrame frame="1" />
        </AllSpots>
        <AllTracks />
        <FilteredTracks />
    </Model>
    <Settings>
        <ImageData filename="Example" folder="" width="512" height="512" nslices="1" nframes="2" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0" />
        <BasicSettings xstart="0" xend="511" ystart="0" yend="511" zstart="0" zend="0" tstart="0" tend="1" />
        <DetectorSettings DETECTOR_NAME="LOG_DETECTOR" TARGET_CHANNEL="1" RADIUS="5.0" THRESHOLD="0.8" DO_MEDIAN_FILTERING="false" DO_SUBPIXEL_LOCALIZATION="true" />
        <InitialSpotFilter feature="QUALITY" value="0.22185452133476247" isabove="true" />
        <SpotFilterCollection />
        <TrackerSettings TRACKER_NAME="SIMPLE_SPARSE_LAP_TRACKER" CUTOFF_PERCENTILE="0.9" ALTERNATIVE_LINKING_COST_FACTOR="1.05" BLOCKING_VALUE="Infinity">
        <Linking LINKING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </Linking>
        <GapClosing ALLOW_GAP_CLOSING="true" GAP_CLOSING_MAX_DISTANCE="15.0" MAX_FRAME_GAP="2">
            <FeaturePenalties />
        </GapClosing>
        <TrackSplitting ALLOW_TRACK_SPLITTING="false" SPLITTING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </TrackSplitting>
        <TrackMerging ALLOW_TRACK_MERGING="false" MERGING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </TrackMerging>
        </TrackerSettings>
        <TrackFilterCollection />
        <AnalyzerCollection>
        <SpotAnalyzers>
            <Analyzer key="Manual spot color" />
            <Analyzer key="Spot intensity" />
            <Analyzer key="EXTRACK_PROBABILITIES" />
            <Analyzer key="Spot contrast and SNR" />
        </SpotAnalyzers>
        <EdgeAnalyzers>
            <Analyzer key="Directional change" />
            <Analyzer key="Edge speed" />
            <Analyzer key="Edge target" />
            <Analyzer key="Edge location" />
            <Analyzer key="Manual edge color" />
        </EdgeAnalyzers>
        <TrackAnalyzers>
            <Analyzer key="Branching analyzer" />
            <Analyzer key="Track duration" />
            <Analyzer key="Track index" />
            <Analyzer key="Track location" />
            <Analyzer key="Track speed" />
            <Analyzer key="Track quality" />
            <Analyzer key="Track motility analysis" />
        </TrackAnalyzers>
        </AnalyzerCollection>
    </Settings>
    <GUIState state="TrackFilter" />
    <DisplaySettings>{
    "name": "CurrentDisplaySettings",
    "spotUniformColor": "204, 51, 204, 255",
    "spotColorByType": "DEFAULT",
    "spotColorByFeature": "UNIFORM_COLOR",
    "spotDisplayRadius": 1.0,
    "spotDisplayedAsRoi": true,
    "spotMin": 0.0,
    "spotMax": 10.0,
    "spotShowName": false,
    "trackMin": 0.0,
    "trackMax": 10.0,
    "trackColorByType": "TRACKS",
    "trackColorByFeature": "TRACK_INDEX",
    "trackUniformColor": "204, 204, 51, 255",
    "undefinedValueColor": "0, 0, 0, 255",
    "missingValueColor": "89, 89, 89, 255",
    "highlightColor": "51, 230, 51, 255",
    "trackDisplayMode": "FULL",
    "colormap": "Jet",
    "limitZDrawingDepth": false,
    "drawingZDepth": 10.0,
    "fadeTracks": true,
    "fadeTrackRange": 30,
    "useAntialiasing": true,
    "spotVisible": true,
    "trackVisible": true,
    "font": {
        "name": "Arial",
        "style": 1,
        "size": 12,
        "pointSize": 12.0,
        "fontSerializedDataVersion": 1
    },
    "lineThickness": 1.0,
    "selectionLineThickness": 4.0,
    "trackschemeBackgroundColor1": "128, 128, 128, 255",
    "trackschemeBackgroundColor2": "192, 192, 192, 255",
    "trackschemeForegroundColor": "0, 0, 0, 255",
    "trackschemeDecorationColor": "0, 0, 0, 255",
    "trackschemeFillBox": false,
    "spotFilled": false,
    "spotTransparencyAlpha": 1.0
    }</DisplaySettings>
    </TrackMate>"""

    # write file
    _write_to_file(path, xml)

    return path


@pytest.fixture
def trackmate_xml(tmp_path) -> Path:
    """Save to the disk a TrackMate XML containing two tracks across two channels."""
    # destination path
    path = tmp_path / "TwoTracks.xml"

    # file content corresponding to an empty Trackmate XML file
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <TrackMate version="7.11.1">
    <Log>TrackMate v7.11.1 started on:
    Tue, 12 Sep 2023 17:55:03
    Please note that TrackMate is available through Fiji, and is based on a publication. If you use it successfully for your research please be so kind to cite our work:
    Ershov, D., Phan, MS., Pylvänäinen, J.W., Rigaud S.U., et al. TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines. Nat Methods (2022). https://doi.org/10.1038/s41592-022-01507-1
    https://doi.org/10.1038/s41592-022-01507-1
    and / or:
    Tinevez, JY.; Perry, N. &amp; Schindelin, J. et al. (2017), 'TrackMate: An open and extensible platform for single-particle tracking.', Methods 115: 80-90, PMID 27713081.
    https://www.sciencedirect.com/science/article/pii/S1046202316303346

    Numerical feature analyzers:
    Spot feature analyzers:
    - Manual spot color provides: Spot color; is manual.
    - Spot intensity provides: Mean ch1, Median ch1, Min ch1, Max ch1, Sum ch1, Std ch1, Mean ch2, Median ch2, Min ch2, Max ch2, Sum ch2, Std ch2.
    - ExTrack probabilities provides: P stuck, P diffusive; is manual.
    - Spot contrast and SNR provides: Ctrst ch1, SNR ch1, Ctrst ch2, SNR ch2.
    - Spot fit 2D ellipse provides: El. x0, El. y0, El. long axis, El. sh. axis, El. angle, El. a.r.
    - Spot 2D shape descriptors provides: Area, Perim., Circ., Solidity, Shape index.
    Edge feature analyzers:
    - Directional change provides: y rate.
    - Edge speed provides: Speed, Disp.
    - Edge target provides: Source ID, Target ID, Cost.
    - Edge location provides: Edge T, Edge X, Edge Y, Edge Z.
    - Manual edge color provides: Edge color; is manual.
    Track feature analyzers:
    - Branching analyzer provides: N spots, N gaps, N splits, N merges, N complex, Lgst gap.
    - Track duration provides: Duration, Track start, Track stop, Track disp.
    - Track index provides: Index, ID.
    - Track location provides: Track X, Track Y, Track Z.
    - Track speed provides: Mean sp., Max speed, Min speed, Med. speed, Std speed.
    - Track quality provides: Mean Q.
    - Track motility analysis provides: Total dist., Max dist., Cfn. ratio, Mn. v. line, Fwd. progr., Mn. y rate.

    Image region of interest:
    Image data:
    For the image named: Merged.tif.
    Matching file Merged.tif in folder: /Users/joran.deschamps/Desktop/
    Geometry:
    X =    0 -  127, dx = 1.00000
    Y =    0 -  127, dy = 1.00000
    Z =    0 -    0, dz = 1.00000
    T =    0 -    1, dt = 1.00000

    Configured detector Thresholding detector with settings:
    - target channel: 1
    - simplify contours: true
    - intensity threshold: 50.0

    Starting detection process using 16 threads.
    Detection processes 2 frames simultaneously and allocates 8 threads per frame.
    Found 4 spots.
    Detection done in 0.0 s.

    Computing spot quality histogram...
    Initial thresholding with a quality threshold above 0.2 ...
    Starting initial filtering process.
    Retained 4 spots out of 4.

    Adding morphology analyzers...
    - Spot fit 2D ellipse provides: El. x0, El. y0, El. long axis, El. sh. axis, El. angle, El. a.r.
    - Spot 2D shape descriptors provides: Area, Perim., Circ., Solidity, Shape index.

    Calculating spot features...
    Calculating features done in 0.0 s.

    Performing spot filtering on the following features:
    No feature threshold set, kept the 4 spots.

    Configured tracker Simple LAP tracker with settings:
    - max frame gap: 2
    - alternative linking cost factor: 1.05
    - linking feature penalties:
    - linking max distance: 15.0
    - gap closing max distance: 15.0
    - merging feature penalties:
    - splitting max distance: 15.0
    - blocking value: Infinity
    - allow gap closing: true
    - allow track splitting: false
    - allow track merging: false
    - merging max distance: 15.0
    - splitting feature penalties:
    - cutoff percentile: 0.9
    - gap closing feature penalties:

    Starting tracking process.
    Tracking done in 0.0 s.
    Found 2 tracks.
    - avg size: 2.0 spots.
    - min size: 2 spots.
    - max size: 2 spots.

    Calculating features done in 0.0 s.

    Performing track filtering on the following features:
    No feature threshold set, kept the 2 tracks.
    Saving data...
    Computing edge features:
    - Directional change in 1 ms.
    - Edge speed in 0 ms.
    - Edge target in 0 ms.
    - Edge location in 0 ms.
    Computation done in 1 ms.
    Computing track features:
    - Branching analyzer in 0 ms.
    - Track duration in 0 ms.
    - Track index in 0 ms.
    - Track location in 0 ms.
    - Track speed in 0 ms.
    - Track quality in 1 ms.
    - Track motility analysis in 0 ms.
    Computation done in 2 ms.
    Added log.
    Added spot, edge and track feature declarations.
    Added 4 spots.
    Added tracks.
    Added filtered tracks.
    Added image information.
    Added crop settings.
    Added detector settings.
    Added initial spot filter.
    Added spot feature filters.
    Added tracker settings.
    Added track feature filters.
    Added spot, edge and track analyzers.
    Added GUI current state.
    Added display settings.
    Writing to file.
    Data saved to: /Users/joran.deschamps/Desktop/Merged.xml
    Saving data...
    Computing edge features:
    - Directional change in 1 ms.
    - Edge speed in 1 ms.
    - Edge target in 0 ms.
    - Edge location in 1 ms.
    Computation done in 3 ms.
    Computing track features:
    - Branching analyzer in 0 ms.
    - Track duration in 1 ms.
    - Track index in 0 ms.
    - Track location in 0 ms.
    - Track speed in 0 ms.
    - Track quality in 1 ms.
    - Track motility analysis in 0 ms.
    Computation done in 2 ms.</Log>
    <Model spatialunits="pixel" timeunits="frame">
        <FeatureDeclarations>
        <SpotFeatures>
            <Feature feature="QUALITY" name="Quality" shortname="Quality" dimension="QUALITY" isint="false" />
            <Feature feature="POSITION_X" name="X" shortname="X" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_Y" name="Y" shortname="Y" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_Z" name="Z" shortname="Z" dimension="POSITION" isint="false" />
            <Feature feature="POSITION_T" name="T" shortname="T" dimension="TIME" isint="false" />
            <Feature feature="FRAME" name="Frame" shortname="Frame" dimension="NONE" isint="true" />
            <Feature feature="RADIUS" name="Radius" shortname="R" dimension="LENGTH" isint="false" />
            <Feature feature="VISIBILITY" name="Visibility" shortname="Visibility" dimension="NONE" isint="true" />
            <Feature feature="MANUAL_SPOT_COLOR" name="Manual spot color" shortname="Spot color" dimension="NONE" isint="true" />
            <Feature feature="MEAN_INTENSITY_CH1" name="Mean intensity ch1" shortname="Mean ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MEDIAN_INTENSITY_CH1" name="Median intensity ch1" shortname="Median ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MIN_INTENSITY_CH1" name="Min intensity ch1" shortname="Min ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MAX_INTENSITY_CH1" name="Max intensity ch1" shortname="Max ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="TOTAL_INTENSITY_CH1" name="Sum intensity ch1" shortname="Sum ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="STD_INTENSITY_CH1" name="Std intensity ch1" shortname="Std ch1" dimension="INTENSITY" isint="false" />
            <Feature feature="MEAN_INTENSITY_CH2" name="Mean intensity ch2" shortname="Mean ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="MEDIAN_INTENSITY_CH2" name="Median intensity ch2" shortname="Median ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="MIN_INTENSITY_CH2" name="Min intensity ch2" shortname="Min ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="MAX_INTENSITY_CH2" name="Max intensity ch2" shortname="Max ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="TOTAL_INTENSITY_CH2" name="Sum intensity ch2" shortname="Sum ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="STD_INTENSITY_CH2" name="Std intensity ch2" shortname="Std ch2" dimension="INTENSITY" isint="false" />
            <Feature feature="EXTRACK_P_STUCK" name="Probability stuck" shortname="P stuck" dimension="NONE" isint="false" />
            <Feature feature="EXTRACK_P_DIFFUSIVE" name="Probability diffusive" shortname="P diffusive" dimension="NONE" isint="false" />
            <Feature feature="CONTRAST_CH1" name="Contrast ch1" shortname="Ctrst ch1" dimension="NONE" isint="false" />
            <Feature feature="SNR_CH1" name="Signal/Noise ratio ch1" shortname="SNR ch1" dimension="NONE" isint="false" />
            <Feature feature="CONTRAST_CH2" name="Contrast ch2" shortname="Ctrst ch2" dimension="NONE" isint="false" />
            <Feature feature="SNR_CH2" name="Signal/Noise ratio ch2" shortname="SNR ch2" dimension="NONE" isint="false" />
            <Feature feature="ELLIPSE_X0" name="Ellipse center x0" shortname="El. x0" dimension="LENGTH" isint="false" />
            <Feature feature="ELLIPSE_Y0" name="Ellipse center y0" shortname="El. y0" dimension="LENGTH" isint="false" />
            <Feature feature="ELLIPSE_MAJOR" name="Ellipse long axis" shortname="El. long axis" dimension="LENGTH" isint="false" />
            <Feature feature="ELLIPSE_MINOR" name="Ellipse short axis" shortname="El. sh. axis" dimension="LENGTH" isint="false" />
            <Feature feature="ELLIPSE_THETA" name="Ellipse angle" shortname="El. angle" dimension="ANGLE" isint="false" />
            <Feature feature="ELLIPSE_ASPECTRATIO" name="Ellipse aspect ratio" shortname="El. a.r." dimension="NONE" isint="false" />
            <Feature feature="AREA" name="Area" shortname="Area" dimension="AREA" isint="false" />
            <Feature feature="PERIMETER" name="Perimeter" shortname="Perim." dimension="LENGTH" isint="false" />
            <Feature feature="CIRCULARITY" name="Circularity" shortname="Circ." dimension="NONE" isint="false" />
            <Feature feature="SOLIDITY" name="Solidity" shortname="Solidity" dimension="NONE" isint="false" />
            <Feature feature="SHAPE_INDEX" name="Shape index" shortname="Shape index" dimension="NONE" isint="false" />
        </SpotFeatures>
        <EdgeFeatures>
            <Feature feature="SPOT_SOURCE_ID" name="Source spot ID" shortname="Source ID" dimension="NONE" isint="true" />
            <Feature feature="SPOT_TARGET_ID" name="Target spot ID" shortname="Target ID" dimension="NONE" isint="true" />
            <Feature feature="LINK_COST" name="Edge cost" shortname="Cost" dimension="COST" isint="false" />
            <Feature feature="DIRECTIONAL_CHANGE_RATE" name="Directional change rate" shortname="y rate" dimension="ANGLE_RATE" isint="false" />
            <Feature feature="SPEED" name="Speed" shortname="Speed" dimension="VELOCITY" isint="false" />
            <Feature feature="DISPLACEMENT" name="Displacement" shortname="Disp." dimension="LENGTH" isint="false" />
            <Feature feature="EDGE_TIME" name="Edge time" shortname="Edge T" dimension="TIME" isint="false" />
            <Feature feature="EDGE_X_LOCATION" name="Edge X" shortname="Edge X" dimension="POSITION" isint="false" />
            <Feature feature="EDGE_Y_LOCATION" name="Edge Y" shortname="Edge Y" dimension="POSITION" isint="false" />
            <Feature feature="EDGE_Z_LOCATION" name="Edge Z" shortname="Edge Z" dimension="POSITION" isint="false" />
            <Feature feature="MANUAL_EDGE_COLOR" name="Manual edge color" shortname="Edge color" dimension="NONE" isint="true" />
        </EdgeFeatures>
        <TrackFeatures>
            <Feature feature="TRACK_INDEX" name="Track index" shortname="Index" dimension="NONE" isint="true" />
            <Feature feature="TRACK_ID" name="Track ID" shortname="ID" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_SPOTS" name="Number of spots in track" shortname="N spots" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_GAPS" name="Number of gaps" shortname="N gaps" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_SPLITS" name="Number of split events" shortname="N splits" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_MERGES" name="Number of merge events" shortname="N merges" dimension="NONE" isint="true" />
            <Feature feature="NUMBER_COMPLEX" name="Number of complex points" shortname="N complex" dimension="NONE" isint="true" />
            <Feature feature="LONGEST_GAP" name="Longest gap" shortname="Lgst gap" dimension="NONE" isint="true" />
            <Feature feature="TRACK_DURATION" name="Track duration" shortname="Duration" dimension="TIME" isint="false" />
            <Feature feature="TRACK_START" name="Track start" shortname="Track start" dimension="TIME" isint="false" />
            <Feature feature="TRACK_STOP" name="Track stop" shortname="Track stop" dimension="TIME" isint="false" />
            <Feature feature="TRACK_DISPLACEMENT" name="Track displacement" shortname="Track disp." dimension="LENGTH" isint="false" />
            <Feature feature="TRACK_X_LOCATION" name="Track mean X" shortname="Track X" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_Y_LOCATION" name="Track mean Y" shortname="Track Y" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_Z_LOCATION" name="Track mean Z" shortname="Track Z" dimension="POSITION" isint="false" />
            <Feature feature="TRACK_MEAN_SPEED" name="Track mean speed" shortname="Mean sp." dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MAX_SPEED" name="Track max speed" shortname="Max speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MIN_SPEED" name="Track min speed" shortname="Min speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MEDIAN_SPEED" name="Track median speed" shortname="Med. speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_STD_SPEED" name="Track std speed" shortname="Std speed" dimension="VELOCITY" isint="false" />
            <Feature feature="TRACK_MEAN_QUALITY" name="Track mean quality" shortname="Mean Q" dimension="QUALITY" isint="false" />
            <Feature feature="TOTAL_DISTANCE_TRAVELED" name="Total distance traveled" shortname="Total dist." dimension="LENGTH" isint="false" />
            <Feature feature="MAX_DISTANCE_TRAVELED" name="Max distance traveled" shortname="Max dist." dimension="LENGTH" isint="false" />
            <Feature feature="CONFINEMENT_RATIO" name="Confinement ratio" shortname="Cfn. ratio" dimension="NONE" isint="false" />
            <Feature feature="MEAN_STRAIGHT_LINE_SPEED" name="Mean straight line speed" shortname="Mn. v. line" dimension="VELOCITY" isint="false" />
            <Feature feature="LINEARITY_OF_FORWARD_PROGRESSION" name="Linearity of forward progression" shortname="Fwd. progr." dimension="NONE" isint="false" />
            <Feature feature="MEAN_DIRECTIONAL_CHANGE_RATE" name="Mean directional change rate" shortname="Mn. y rate" dimension="ANGLE_RATE" isint="false" />
        </TrackFeatures>
        </FeatureDeclarations>
        <AllSpots nspots="4">
        <SpotsInFrame frame="0">
            <Spot ID="336349" name="ID336349" STD_INTENSITY_CH1="8.931232686098406" SOLIDITY="0.9850746268656716" STD_INTENSITY_CH2="18.902997335625752" QUALITY="132.0" POSITION_T="0.0" TOTAL_INTENSITY_CH2="28776.0" TOTAL_INTENSITY_CH1="13596.0" CONTRAST_CH1="0.9748757957572208" ELLIPSE_MINOR="6.512137745815348" ELLIPSE_THETA="1.029328123984239" ELLIPSE_Y0="-0.025614756683639262" FRAME="0" CIRCULARITY="0.9446283675768872" AREA="132.0" ELLIPSE_MAJOR="6.6123899768885375" CONTRAST_CH2="0.9748757957572208" MEAN_INTENSITY_CH1="102.22556390977444" MAX_INTENSITY_CH2="218.0" MEAN_INTENSITY_CH2="216.3609022556391" MAX_INTENSITY_CH1="103.0" MIN_INTENSITY_CH2="0.0" MIN_INTENSITY_CH1="0.0" SNR_CH1="11.300238433644296" ELLIPSE_X0="-0.03394515689449156" SHAPE_INDEX="3.647324805297581" SNR_CH2="11.300238433644296" MEDIAN_INTENSITY_CH1="103.0" VISIBILITY="1" RADIUS="6.4820448144285745" MEDIAN_INTENSITY_CH2="218.0" POSITION_X="78.27020202020202" POSITION_Y="75.06186868686868" ELLIPSE_ASPECTRATIO="1.0153946729916163" POSITION_Z="0.0" PERIMETER="41.90457167260814" ROI_N_POINTS="19">2.2297979797979792 6.438131313131322 4.229797979797979 5.438131313131322 5.229797979797979 3.438131313131322 6.229797979797979 2.438131313131322 6.229797979797979 0.43813131313132203 6.229797979797979 -1.561868686868678 5.229797979797979 -3.561868686868678 4.229797979797979 -5.561868686868678 0.22979797979797922 -6.561868686868678 -1.7702020202020208 -6.561868686868678 -2.7702020202020208 -5.561868686868678 -4.770202020202021 -4.561868686868678 -6.770202020202021 -0.561868686868678 -6.770202020202021 1.438131313131322 -5.770202020202021 2.438131313131322 -4.770202020202021 4.438131313131322 -2.7702020202020208 5.438131313131322 -1.7702020202020208 6.438131313131322 0.22979797979797922 6.438131313131322</Spot>
            <Spot ID="336347" name="ID336347" STD_INTENSITY_CH1="21.417616247245707" SOLIDITY="0.9850746268656716" STD_INTENSITY_CH2="8.931232686098406" QUALITY="132.0" POSITION_T="0.0" TOTAL_INTENSITY_CH2="13596.0" TOTAL_INTENSITY_CH1="32604.0" CONTRAST_CH1="0.9748757957572208" ELLIPSE_MINOR="6.512137745815348" ELLIPSE_THETA="1.029328123984246" ELLIPSE_Y0="-0.02561475668365407" FRAME="0" CIRCULARITY="0.9446283675768872" AREA="132.0" ELLIPSE_MAJOR="6.612389976888538" CONTRAST_CH2="0.9748757957572208" MEAN_INTENSITY_CH1="245.14285714285714" MAX_INTENSITY_CH2="103.0" MEAN_INTENSITY_CH2="102.22556390977444" MAX_INTENSITY_CH1="247.0" MIN_INTENSITY_CH2="0.0" MIN_INTENSITY_CH1="0.0" SNR_CH1="11.300238433644287" ELLIPSE_X0="-0.03394515689449155" SHAPE_INDEX="3.647324805297581" SNR_CH2="11.300238433644296" MEDIAN_INTENSITY_CH1="247.0" VISIBILITY="1" RADIUS="6.4820448144285745" MEDIAN_INTENSITY_CH2="103.0" POSITION_X="40.27020202020202" POSITION_Y="42.06186868686869" ELLIPSE_ASPECTRATIO="1.0153946729916166" POSITION_Z="0.0" PERIMETER="41.90457167260814" ROI_N_POINTS="19">2.2297979797979792 6.438131313131308 4.229797979797979 5.438131313131308 5.229797979797979 3.438131313131308 6.229797979797979 2.438131313131308 6.229797979797979 0.4381313131313078 6.229797979797979 -1.5618686868686922 5.229797979797979 -3.561868686868692 4.229797979797979 -5.561868686868692 0.22979797979797922 -6.561868686868692 -1.7702020202020208 -6.561868686868692 -2.7702020202020208 -5.561868686868692 -4.770202020202021 -4.561868686868692 -6.770202020202021 -0.5618686868686922 -6.770202020202021 1.4381313131313078 -5.770202020202021 2.438131313131308 -4.770202020202021 4.438131313131308 -2.7702020202020208 5.438131313131308 -1.7702020202020208 6.438131313131308 0.22979797979797922 6.438131313131308</Spot>
        </SpotsInFrame>
        <SpotsInFrame frame="1">
            <Spot ID="336348" name="ID336348" STD_INTENSITY_CH1="20.550506277721592" SOLIDITY="0.9850746268656716" STD_INTENSITY_CH2="16.995355402672857" QUALITY="132.0" POSITION_T="1.0" TOTAL_INTENSITY_CH2="25872.0" TOTAL_INTENSITY_CH1="31284.0" CONTRAST_CH1="0.9748757957572208" ELLIPSE_MINOR="6.512137745815348" ELLIPSE_THETA="1.029328123984246" ELLIPSE_Y0="-0.02561475668365407" FRAME="1" CIRCULARITY="0.9446283675768872" AREA="132.0" ELLIPSE_MAJOR="6.612389976888538" CONTRAST_CH2="0.9748757957572207" MEAN_INTENSITY_CH1="235.21804511278197" MAX_INTENSITY_CH2="196.0" MEAN_INTENSITY_CH2="194.52631578947367" MAX_INTENSITY_CH1="237.0" MIN_INTENSITY_CH2="0.0" MIN_INTENSITY_CH1="0.0" SNR_CH1="11.300238433644287" ELLIPSE_X0="-0.03394515689449155" SHAPE_INDEX="3.647324805297581" SNR_CH2="11.300238433644186" MEDIAN_INTENSITY_CH1="237.0" VISIBILITY="1" RADIUS="6.4820448144285745" MEDIAN_INTENSITY_CH2="196.0" POSITION_X="40.27020202020202" POSITION_Y="39.06186868686869" ELLIPSE_ASPECTRATIO="1.0153946729916166" POSITION_Z="0.0" PERIMETER="41.90457167260814" ROI_N_POINTS="19">2.2297979797979792 6.438131313131308 4.229797979797979 5.438131313131308 5.229797979797979 3.438131313131308 6.229797979797979 2.438131313131308 6.229797979797979 0.4381313131313078 6.229797979797979 -1.5618686868686922 5.229797979797979 -3.561868686868692 4.229797979797979 -5.561868686868692 0.22979797979797922 -6.561868686868692 -1.7702020202020208 -6.561868686868692 -2.7702020202020208 -5.561868686868692 -4.770202020202021 -4.561868686868692 -6.770202020202021 -0.5618686868686922 -6.770202020202021 1.4381313131313078 -5.770202020202021 2.438131313131308 -4.770202020202021 4.438131313131308 -2.7702020202020208 5.438131313131308 -1.7702020202020208 6.438131313131308 0.22979797979797922 6.438131313131308</Spot>
            <Spot ID="336350" name="ID336350" STD_INTENSITY_CH1="15.954823439243714" SOLIDITY="0.9850746268656716" STD_INTENSITY_CH2="8.757810692193601" QUALITY="132.0" POSITION_T="1.0" TOTAL_INTENSITY_CH2="13332.0" TOTAL_INTENSITY_CH1="24288.0" CONTRAST_CH1="0.9748757957572207" ELLIPSE_MINOR="6.512137745815348" ELLIPSE_THETA="1.029328123984239" ELLIPSE_Y0="-0.025614756683639262" FRAME="1" CIRCULARITY="0.9446283675768872" AREA="132.0" ELLIPSE_MAJOR="6.6123899768885375" CONTRAST_CH2="0.9748757957572208" MEAN_INTENSITY_CH1="182.61654135338347" MAX_INTENSITY_CH2="101.0" MEAN_INTENSITY_CH2="100.2406015037594" MAX_INTENSITY_CH1="184.0" MIN_INTENSITY_CH2="0.0" MIN_INTENSITY_CH1="0.0" SNR_CH1="11.300238433644324" ELLIPSE_X0="-0.03394515689449156" SHAPE_INDEX="3.647324805297581" SNR_CH2="11.300238433644271" MEDIAN_INTENSITY_CH1="184.0" VISIBILITY="1" RADIUS="6.4820448144285745" MEDIAN_INTENSITY_CH2="101.0" POSITION_X="77.27020202020202" POSITION_Y="77.06186868686868" ELLIPSE_ASPECTRATIO="1.0153946729916163" POSITION_Z="0.0" PERIMETER="41.90457167260814" ROI_N_POINTS="19">2.2297979797979792 6.438131313131322 4.229797979797979 5.438131313131322 5.229797979797979 3.438131313131322 6.229797979797979 2.438131313131322 6.229797979797979 0.43813131313132203 6.229797979797979 -1.561868686868678 5.229797979797979 -3.561868686868678 4.229797979797979 -5.561868686868678 0.22979797979797922 -6.561868686868678 -1.7702020202020208 -6.561868686868678 -2.7702020202020208 -5.561868686868678 -4.770202020202021 -4.561868686868678 -6.770202020202021 -0.561868686868678 -6.770202020202021 1.438131313131322 -5.770202020202021 2.438131313131322 -4.770202020202021 4.438131313131322 -2.7702020202020208 5.438131313131322 -1.7702020202020208 6.438131313131322 0.22979797979797922 6.438131313131322</Spot>
        </SpotsInFrame>
        </AllSpots>
        <AllTracks>
        <Track name="Track_0" TRACK_ID="0" TRACK_INDEX="0" NUMBER_SPOTS="2" NUMBER_GAPS="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0" LONGEST_GAP="0" TRACK_DURATION="1.0" TRACK_START="0.0" TRACK_STOP="1.0" TRACK_DISPLACEMENT="2.23606797749979" TRACK_X_LOCATION="77.77020202020202" TRACK_Y_LOCATION="76.06186868686868" TRACK_Z_LOCATION="0.0" TRACK_MEAN_SPEED="2.23606797749979" TRACK_MAX_SPEED="2.23606797749979" TRACK_MIN_SPEED="2.23606797749979" TRACK_MEDIAN_SPEED="2.23606797749979" TRACK_STD_SPEED="NaN" TRACK_MEAN_QUALITY="132.0" TOTAL_DISTANCE_TRAVELED="2.23606797749979" MAX_DISTANCE_TRAVELED="2.23606797749979" CONFINEMENT_RATIO="1.0" MEAN_STRAIGHT_LINE_SPEED="2.23606797749979" LINEARITY_OF_FORWARD_PROGRESSION="1.0" MEAN_DIRECTIONAL_CHANGE_RATE="NaN">
            <Edge SPOT_SOURCE_ID="336349" SPOT_TARGET_ID="336350" LINK_COST="5.0" DIRECTIONAL_CHANGE_RATE="NaN" SPEED="2.23606797749979" DISPLACEMENT="2.23606797749979" EDGE_TIME="0.5" EDGE_X_LOCATION="77.77020202020202" EDGE_Y_LOCATION="76.06186868686868" EDGE_Z_LOCATION="0.0" />
        </Track>
        <Track name="Track_1" TRACK_ID="1" TRACK_INDEX="1" NUMBER_SPOTS="2" NUMBER_GAPS="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0" LONGEST_GAP="0" TRACK_DURATION="1.0" TRACK_START="0.0" TRACK_STOP="1.0" TRACK_DISPLACEMENT="3.0" TRACK_X_LOCATION="40.27020202020202" TRACK_Y_LOCATION="40.56186868686869" TRACK_Z_LOCATION="0.0" TRACK_MEAN_SPEED="3.0" TRACK_MAX_SPEED="3.0" TRACK_MIN_SPEED="3.0" TRACK_MEDIAN_SPEED="3.0" TRACK_STD_SPEED="NaN" TRACK_MEAN_QUALITY="132.0" TOTAL_DISTANCE_TRAVELED="3.0" MAX_DISTANCE_TRAVELED="3.0" CONFINEMENT_RATIO="1.0" MEAN_STRAIGHT_LINE_SPEED="3.0" LINEARITY_OF_FORWARD_PROGRESSION="1.0" MEAN_DIRECTIONAL_CHANGE_RATE="NaN">
            <Edge SPOT_SOURCE_ID="336347" SPOT_TARGET_ID="336348" LINK_COST="9.0" DIRECTIONAL_CHANGE_RATE="NaN" SPEED="3.0" DISPLACEMENT="3.0" EDGE_TIME="0.5" EDGE_X_LOCATION="40.27020202020202" EDGE_Y_LOCATION="40.56186868686869" EDGE_Z_LOCATION="0.0" />
        </Track>
        </AllTracks>
        <FilteredTracks>
        <TrackID TRACK_ID="0" />
        <TrackID TRACK_ID="1" />
        </FilteredTracks>
    </Model>
    <Settings>
        <ImageData filename="Merged.tif" folder="/Users/joran.deschamps/Desktop/" width="128" height="128" nslices="1" nframes="2" pixelwidth="1.0" pixelheight="1.0" voxeldepth="1.0" timeinterval="1.0" />
        <BasicSettings xstart="0" xend="127" ystart="0" yend="127" zstart="0" zend="0" tstart="0" tend="1" />
        <DetectorSettings DETECTOR_NAME="THRESHOLD_DETECTOR" TARGET_CHANNEL="1" INTENSITY_THRESHOLD="50.0" SIMPLIFY_CONTOURS="true" />
        <InitialSpotFilter feature="QUALITY" value="0.22185452133476247" isabove="true" />
        <SpotFilterCollection />
        <TrackerSettings TRACKER_NAME="SIMPLE_SPARSE_LAP_TRACKER" CUTOFF_PERCENTILE="0.9" ALTERNATIVE_LINKING_COST_FACTOR="1.05" BLOCKING_VALUE="Infinity">
        <Linking LINKING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </Linking>
        <GapClosing ALLOW_GAP_CLOSING="true" GAP_CLOSING_MAX_DISTANCE="15.0" MAX_FRAME_GAP="2">
            <FeaturePenalties />
        </GapClosing>
        <TrackSplitting ALLOW_TRACK_SPLITTING="false" SPLITTING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </TrackSplitting>
        <TrackMerging ALLOW_TRACK_MERGING="false" MERGING_MAX_DISTANCE="15.0">
            <FeaturePenalties />
        </TrackMerging>
        </TrackerSettings>
        <TrackFilterCollection />
        <AnalyzerCollection>
        <SpotAnalyzers>
            <Analyzer key="Manual spot color" />
            <Analyzer key="Spot intensity" />
            <Analyzer key="EXTRACK_PROBABILITIES" />
            <Analyzer key="Spot contrast and SNR" />
            <Analyzer key="Spot fit 2D ellipse" />
            <Analyzer key="Spot 2D shape descriptors" />
        </SpotAnalyzers>
        <EdgeAnalyzers>
            <Analyzer key="Directional change" />
            <Analyzer key="Edge speed" />
            <Analyzer key="Edge target" />
            <Analyzer key="Edge location" />
            <Analyzer key="Manual edge color" />
        </EdgeAnalyzers>
        <TrackAnalyzers>
            <Analyzer key="Branching analyzer" />
            <Analyzer key="Track duration" />
            <Analyzer key="Track index" />
            <Analyzer key="Track location" />
            <Analyzer key="Track speed" />
            <Analyzer key="Track quality" />
            <Analyzer key="Track motility analysis" />
        </TrackAnalyzers>
        </AnalyzerCollection>
    </Settings>
    <GUIState state="ConfigureViews" />
    <DisplaySettings>{
    "name": "CurrentDisplaySettings",
    "spotUniformColor": "204, 51, 204, 255",
    "spotColorByType": "DEFAULT",
    "spotColorByFeature": "UNIFORM_COLOR",
    "spotDisplayRadius": 1.0,
    "spotDisplayedAsRoi": true,
    "spotMin": 0.0,
    "spotMax": 10.0,
    "spotShowName": false,
    "trackMin": 0.0,
    "trackMax": 10.0,
    "trackColorByType": "TRACKS",
    "trackColorByFeature": "TRACK_INDEX",
    "trackUniformColor": "204, 204, 51, 255",
    "undefinedValueColor": "0, 0, 0, 255",
    "missingValueColor": "89, 89, 89, 255",
    "highlightColor": "51, 230, 51, 255",
    "trackDisplayMode": "FULL",
    "colormap": "Jet",
    "limitZDrawingDepth": false,
    "drawingZDepth": 10.0,
    "fadeTracks": true,
    "fadeTrackRange": 30,
    "useAntialiasing": true,
    "spotVisible": true,
    "trackVisible": true,
    "font": {
        "name": "Arial",
        "style": 1,
        "size": 12,
        "pointSize": 12.0,
        "fontSerializedDataVersion": 1
    },
    "lineThickness": 1.0,
    "selectionLineThickness": 4.0,
    "trackschemeBackgroundColor1": "128, 128, 128, 255",
    "trackschemeBackgroundColor2": "192, 192, 192, 255",
    "trackschemeForegroundColor": "0, 0, 0, 255",
    "trackschemeDecorationColor": "0, 0, 0, 255",
    "trackschemeFillBox": false,
    "spotFilled": false,
    "spotTransparencyAlpha": 1.0
    }</DisplaySettings>
    </TrackMate>
    """

    # write file
    _write_to_file(path, xml)

    return path
