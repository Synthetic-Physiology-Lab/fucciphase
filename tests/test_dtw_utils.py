import numpy as np

from fucciphase.utils.dtw import get_time_distortion_coefficient


def test_distortion_as_in_paper():
    """Test distortion path as in Liu et al.

    Notes
    -----

    Reference:

    Liu, Yutao, et al.
    "A novel distance measure based on dynamic time warping to improve time series classification."
    Information Sciences 656 (2024): 119921.
    """
    # Note: correction to Fig. 2 in paper, last entry must be 5 instead of 6
    test_path = [[1, 1], [2, 1], [3, 2], [4, 3], [5, 4], [5, 5], [5, 6], [5, 7]]
    expected = [-1, 0, 0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0]
    time_distortion_array, distortion_score, compress_count, stretch_count = (
        get_time_distortion_coefficient(test_path)
    )
    assert np.all(np.isclose(expected, time_distortion_array))
    assert np.isclose(distortion_score, 4)
    assert compress_count == 4
    assert stretch_count == 1


def test_distortion_as_in_paper_numpy():
    """Test distortion path as in Liu et al.

    Notes
    -----

    Reference:

    Liu, Yutao, et al.
    "A novel distance measure based on dynamic time warping to improve time series classification."
    Information Sciences 656 (2024): 119921.
    """
    # Note: correction to Fig. 2 in paper, last entry must be 5 instead of 6
    test_path = np.array(
        [[1, 1], [2, 1], [3, 2], [4, 3], [5, 4], [5, 5], [5, 6], [5, 7]]
    )
    expected = [-1, 0, 0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0]
    time_distortion_array, distortion_score, compress_count, stretch_count = (
        get_time_distortion_coefficient(test_path)
    )
    assert np.all(np.isclose(expected, time_distortion_array))
    assert np.isclose(distortion_score, 4)
    assert compress_count == 4
    assert stretch_count == 1


def test_distortion_as_in_paper_extended():
    """Test distortion path as in Liu et al.

    Notes
    -----

    Reference:

    Liu, Yutao, et al.
    "A novel distance measure based on dynamic time warping to improve time series classification."
    Information Sciences 656 (2024): 119921.
    """
    # Note: correction to Fig. 2 in paper, last entry must be 5 instead of 6
    test_path = [
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 3],
        [5, 4],
        [5, 5],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
    ]
    expected = [-1, 0, 0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 0, 0]
    time_distortion_array, distortion_score, compress_count, stretch_count = (
        get_time_distortion_coefficient(test_path)
    )
    assert np.all(np.isclose(expected, time_distortion_array))
    assert np.isclose(distortion_score, 4)
    assert compress_count == 4
    assert stretch_count == 1


def test_distortion_as_in_paper_end_alpha():
    """Test distortion path as in Liu et al.

    Notes
    -----

    Reference:

    Liu, Yutao, et al.
    "A novel distance measure based on dynamic time warping to improve time series classification."
    Information Sciences 656 (2024): 119921.
    """
    # Note: correction to Fig. 2 in paper, last entry must be 5 instead of 6
    test_path = [
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 3],
        [5, 4],
        [5, 5],
        [5, 6],
        [5, 7],
        [6, 7],
        [7, 7],
    ]
    expected = [-1, 0, 0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0, -2, -2]
    time_distortion_array, distortion_score, compress_count, stretch_count = (
        get_time_distortion_coefficient(test_path)
    )
    assert np.all(np.isclose(expected, time_distortion_array))
    assert np.isclose(distortion_score, 8)
    assert compress_count == 4
    assert stretch_count == 3
