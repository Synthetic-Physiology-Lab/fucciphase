"""
Convenience functions for fucciphase - Utility submodule for FUCCIphase.

This package collects helper functions used throughout FUCCIphase, including:
- TrackMate XML parsing and rewriting
- Channel normalization
- Track splitting and postprocessing
- Motility and lineage visualizations
- Synthetic data generation
- DTW time-distortion utilities

Everything listed in ``__all__`` is considered part of the public FUCCIphase API.

"""

__all__ = [
    "TrackMateXML",
    "check_channels",
    "check_thresholds",
    "compute_motility_parameters",
    "export_lineage_tree_to_svg",
    "fit_percentages",
    "get_norm_channel_name",
    "get_time_distortion_coefficient",
    "norm",
    "normalize_channels",
    "plot_trackscheme",
    "postprocess_estimated_percentages",
    "simulate_single_track",
    "split_all_tracks",
    "split_track",
    "split_trackmate_tracks",
]

from .checks import check_channels, check_thresholds
from .dtw import get_time_distortion_coefficient
from .normalize import get_norm_channel_name, norm, normalize_channels
from .phase_fit import fit_percentages, postprocess_estimated_percentages
from .simulator import simulate_single_track
from .track_postprocessing import (
    compute_motility_parameters,
    export_lineage_tree_to_svg,
    plot_trackscheme,
    split_all_tracks,
    split_track,
    split_trackmate_tracks,
)
from .trackmate import TrackMateXML
