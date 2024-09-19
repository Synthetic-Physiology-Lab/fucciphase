import pandas as pd


def get_feature_value_at_frame(
    labels: pd.DataFrame, label_name: str, label: int, feature: str
) -> float:
    """Helper function to get value of feature."""
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
    """Prepare a DF with penalties for tracking.

    Notes
    -----
    See more details here:
    https://laptrack.readthedocs.io/en/stable/examples/custom_metric.html

    The penalty formulation is similar to TrackMate,
    see
    https://imagej.net/plugins/trackmate/trackers/lap-trackers#calculating-linking-costs
    The penalty is computed as:
    P = 1 + sum(feature_penalties)
    Each feature penalty is:
    p = 3 * weight * abs(f1 - f2) / (f1 + f2)

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
