"""Tests for the phase module, focusing on signal mode functionality."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fucciphase.phase import (
    NewColumns,
    _compute_output_length_offset,
    _process_channel,
    estimate_percentage_by_subsequence_alignment,
)


class TestProcessChannel:
    """Tests for the _process_channel helper function."""

    def test_signal_mode_returns_single_array(self):
        """Signal mode should return only the signal."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _process_channel(series, "signal", smooth=0.1)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], series)

    def test_derivative_mode_returns_single_array(self):
        """Derivative mode should return only the derivative."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _process_channel(series, "derivative", smooth=0.1)

        assert len(result) == 1
        # Derivative reduces length by 1
        assert len(result[0]) == len(series) - 1

    def test_both_mode_returns_two_arrays(self):
        """Both mode should return signal and derivative."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _process_channel(series, "both", smooth=0.1)

        assert len(result) == 2
        # First is signal (same length)
        assert len(result[0]) == len(series)
        # Second is derivative (length - 1)
        assert len(result[1]) == len(series) - 1

    def test_signal_mode_preserves_original(self):
        """Signal mode should not modify the original array."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        original = series.copy()
        _process_channel(series, "signal", smooth=0.1)

        np.testing.assert_array_equal(series, original)


class TestComputeOutputLengthOffset:
    """Tests for the _compute_output_length_offset helper function."""

    def test_signal_mode_returns_zero(self):
        """Signal mode should return offset 0."""
        assert _compute_output_length_offset("signal") == 0

    def test_derivative_mode_returns_one(self):
        """Derivative mode should return offset 1."""
        assert _compute_output_length_offset("derivative") == 1

    def test_both_mode_returns_one(self):
        """Both mode should return offset 1 (derivative is used)."""
        assert _compute_output_length_offset("both") == 1


@pytest.fixture
def reference_data():
    """Create reference data for DTW alignment tests."""
    # Create a simple reference curve covering one cell cycle
    time = np.linspace(0, 24, 100)  # 24 hours
    percentage = np.linspace(0, 100, 100)

    # Simple sinusoidal patterns for channels
    ch1 = 0.5 + 0.5 * np.sin(2 * np.pi * time / 24)
    ch2 = 0.5 + 0.5 * np.cos(2 * np.pi * time / 24)

    return pd.DataFrame(
        {"time": time, "percentage": percentage, "CH1": ch1, "CH2": ch2}
    )


@pytest.fixture
def track_data():
    """Create track data for DTW alignment tests."""
    # Create a track with 20 frames
    n_frames = 20
    frames = np.arange(n_frames)

    # Similar patterns to reference
    ch1 = 0.5 + 0.5 * np.sin(2 * np.pi * frames / 20)
    ch2 = 0.5 + 0.5 * np.cos(2 * np.pi * frames / 20)

    return pd.DataFrame(
        {"FRAME": frames, "TRACK_ID": [1] * n_frames, "CH1": ch1, "CH2": ch2}
    )


class TestEstimatePercentageBySubsequenceAlignment:
    """Tests for the estimate_percentage_by_subsequence_alignment function."""

    def test_signal_mode_produces_output(self, reference_data, track_data):
        """Signal mode should produce cell cycle percentage estimates."""
        estimate_percentage_by_subsequence_alignment(
            track_data,
            dt=0.25,
            channels=["CH1", "CH2"],
            reference_data=reference_data,
            signal_mode="signal",
        )

        assert NewColumns.cell_cycle_dtw() in track_data.columns
        # Should have values (not all NaN)
        assert not track_data[NewColumns.cell_cycle_dtw()].isna().all()

    def test_derivative_mode_produces_output(self, reference_data, track_data):
        """Derivative mode should produce cell cycle percentage estimates."""
        estimate_percentage_by_subsequence_alignment(
            track_data,
            dt=0.25,
            channels=["CH1", "CH2"],
            reference_data=reference_data,
            signal_mode="derivative",
        )

        assert NewColumns.cell_cycle_dtw() in track_data.columns
        assert not track_data[NewColumns.cell_cycle_dtw()].isna().all()

    def test_both_mode_produces_output(self, reference_data, track_data):
        """Both mode should produce cell cycle percentage estimates."""
        estimate_percentage_by_subsequence_alignment(
            track_data,
            dt=0.25,
            channels=["CH1", "CH2"],
            reference_data=reference_data,
            signal_mode="both",
        )

        assert NewColumns.cell_cycle_dtw() in track_data.columns
        assert not track_data[NewColumns.cell_cycle_dtw()].isna().all()

    def test_backward_compatibility_use_derivative_true(
        self, reference_data, track_data
    ):
        """use_derivative=True should work and emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimate_percentage_by_subsequence_alignment(
                track_data,
                dt=0.25,
                channels=["CH1", "CH2"],
                reference_data=reference_data,
                use_derivative=True,
            )

            # Check deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "use_derivative is deprecated" in str(w[0].message)

        assert NewColumns.cell_cycle_dtw() in track_data.columns

    def test_backward_compatibility_use_derivative_false(
        self, reference_data, track_data
    ):
        """use_derivative=False should work and emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimate_percentage_by_subsequence_alignment(
                track_data,
                dt=0.25,
                channels=["CH1", "CH2"],
                reference_data=reference_data,
                use_derivative=False,
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert NewColumns.cell_cycle_dtw() in track_data.columns

    def test_short_tracks_get_nan(self, reference_data):
        """Tracks shorter than minimum_track_length should get NaN values."""
        # Create a very short track
        short_track = pd.DataFrame(
            {
                "FRAME": [0, 1, 2],
                "TRACK_ID": [1, 1, 1],
                "CH1": [0.5, 0.6, 0.7],
                "CH2": [0.3, 0.4, 0.5],
            }
        )

        estimate_percentage_by_subsequence_alignment(
            short_track,
            dt=0.25,
            channels=["CH1", "CH2"],
            reference_data=reference_data,
            signal_mode="derivative",
            minimum_track_length=10,
        )

        # All values should be NaN for short track
        assert short_track[NewColumns.cell_cycle_dtw()].isna().all()

    def test_missing_time_column_raises(self, track_data):
        """Missing 'time' column in reference data should raise ValueError."""
        bad_reference = pd.DataFrame(
            {"percentage": [0, 50, 100], "CH1": [0.1, 0.5, 0.9], "CH2": [0.9, 0.5, 0.1]}
        )

        with pytest.raises(ValueError, match="time column"):
            estimate_percentage_by_subsequence_alignment(
                track_data,
                dt=0.25,
                channels=["CH1", "CH2"],
                reference_data=bad_reference,
                signal_mode="derivative",
            )

    def test_missing_percentage_column_raises(self, track_data):
        """Missing 'percentage' column in reference data should raise ValueError."""
        bad_reference = pd.DataFrame(
            {"time": [0, 12, 24], "CH1": [0.1, 0.5, 0.9], "CH2": [0.9, 0.5, 0.1]}
        )

        with pytest.raises(ValueError, match="percentage column"):
            estimate_percentage_by_subsequence_alignment(
                track_data,
                dt=0.25,
                channels=["CH1", "CH2"],
                reference_data=bad_reference,
                signal_mode="derivative",
            )

    def test_missing_channel_in_reference_raises(self, track_data):
        """Missing channel in reference data should raise ValueError."""
        bad_reference = pd.DataFrame(
            {
                "time": [0, 12, 24],
                "percentage": [0, 50, 100],
                "CH1": [0.1, 0.5, 0.9],
                # CH2 is missing
            }
        )

        with pytest.raises(ValueError, match="channel names"):
            estimate_percentage_by_subsequence_alignment(
                track_data,
                dt=0.25,
                channels=["CH1", "CH2"],
                reference_data=bad_reference,
                signal_mode="derivative",
            )

    def test_dtw_metrics_are_saved(self, reference_data, track_data):
        """DTW distance and distortion metrics should be saved."""
        estimate_percentage_by_subsequence_alignment(
            track_data,
            dt=0.25,
            channels=["CH1", "CH2"],
            reference_data=reference_data,
            signal_mode="derivative",
        )

        # Check that DTW metrics are present
        assert NewColumns.dtw_distance() in track_data.columns
        assert NewColumns.dtw_distortion() in track_data.columns
        assert NewColumns.dtw_warping_amount() in track_data.columns
