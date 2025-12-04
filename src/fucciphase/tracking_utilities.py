import pandas as pd


def get_feature_value_at_frame(
    labels: pd.DataFrame, label_name: str, label: int, feature: str
) -> float:
    """
    Helper function to get the value of a feature for a given label.

    Parameters
    ----------
    labels : pandas.DataFrame
        Dataframe containing at least the label and feature columns.
    label_name : str
        Column name containing label identifiers.
    label : int
        Label value to select.
    feature : str
        Column name of the feature whose value should be returned.

    Returns
    -------
    float
        The feature value corresponding to the specified label.

    Raises
    ------
    ValueError
        If zero or multiple rows match the requested label.
    """
    value = labels[labels[label_name] == label, feature].to_numpy()
    assert len(value) == 1
    return float(value[0])


def prepare_penalty_df(
    df: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    frame_name: str = "FRAME",
    label_name: str = "LABEL",
    weight: float = 1.0,
) -> pd.DataFrame:
    """Prepare a dataframe with penalties for tracking.

    This function is intended to construct a cost / penalty matrix for
    linking detections between consecutive frames, inspired by the
    formulations used in LapTrack and TrackMate.

    Notes
    -----
    See more details here:
    - https://laptrack.readthedocs.io/en/stable/examples/custom_metric.html
    - https://imagej.net/plugins/trackmate/trackers/lap-trackers#calculating-linking-costs

    The intended penalty formulation is:
    P = 1 + sum(feature_penalties)
    with each feature penalty of the form:
    p = 3 * weight * abs(f1 - f2) / (f1 + f2)

    **Current status:** this function is not yet stably implemented and will
    raise ``NotImplementedError`` if called. The prototype implementation
    below is kept for future development.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least frame, label and feature columns.
    feature_1 : str
        Name of the first feature to use in the penalty.
    feature_2 : str
        Name of the second feature to use in the penalty.
    frame_name : str, optional
        Column name for frame indices. Default is ``"FRAME"``.
    label_name : str, optional
        Column name for object labels. Default is ``"LABEL"``.
    weight : float, optional
        Global scaling factor for feature penalties. Default is 1.0.

    Returns
    -------
    pandas.DataFrame
        Prototype would return a dataframe indexed by (frame, label1, label2)
        with a ``penalty`` column.

    Raises
    ------
    NotImplementedError
        This function is currently not stable and is intentionally disabled.

    """
    raise NotImplementedError("This function is not yet stably implemented.")

    penalty_records = []
    frames = df[frame_name].unique()
    for i, frame in enumerate(frames):
        # skip last frame
        if i == len(frames) - 1:
            continue
        next_frame = frames[i + 1]
        labels = df.loc[df[frame_name] == frame, label_name]
        next_labels = df.loc[df[frame_name] == next_frame, label_name]
        for label in labels:
            if label == 0:
                continue
            # get index where frame + label
            value1 = get_feature_value_at_frame(labels, label_name, label, feature_1)
            value2 = get_feature_value_at_frame(labels, label_name, label, feature_2)
            for next_label in labels:
                if next_label == 0:
                    continue

                next_value1 = get_feature_value_at_frame(
                    next_labels, label_name, label, feature_1
                )
                next_value2 = get_feature_value_at_frame(
                    next_labels, label_name, label, feature_2
                )
                penalty = (
                    3.0 * weight * abs(value1 - next_value1) / (value1 + next_value1)
                )
                penalty += (
                    3.0 * weight * abs(value2 - next_value2) / (value2 + next_value2)
                )
                penalty += 1
                penalty_records.append(
                    {
                        "frame": frame,
                        "label1": label,
                        "label2": next_label,
                        "penalty": penalty,
                    }
                )
    penalty_df = pd.DataFrame.from_records(penalty_records)

    return penalty_df.set_index(["frame", "label1", "label2"]).copy()
