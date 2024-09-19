from typing import List


def check_channels(n_fluorophores: int, channels: List[str]) -> None:
    """Check number of channels."""
    if len(channels) != n_fluorophores:
        raise ValueError(f"Need to provide {n_fluorophores} channel names.")


def check_thresholds(n_fluorophores: int, thresholds: List[float]) -> None:
    """Check correct format and range of thresholds."""
    if len(thresholds) != n_fluorophores:
        raise ValueError("Provide one threshold per channel.")
    # check that the thresholds are between 0 and 1
    if not all(0 < t < 1 for t in thresholds):
        raise ValueError("Thresholds must be between 0 and 1.")
