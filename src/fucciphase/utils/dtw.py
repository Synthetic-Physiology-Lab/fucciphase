from typing import List, Union

import numpy as np


def get_time_distortion_coefficient(
    path: Union[np.ndarray, List[List[float]]],
) -> tuple[np.ndarray, float, int, int]:
    """Compute distortion coefficient from warping path.

    Parameters
    ----------
    path: np.ndarray
        Warping path, first dimension query index, second reference index

    The warping path holds two indices: the index of the query (first entry)
    and the index of the reference curve (second entry)

    """
    lmbd = np.zeros(len(path) - 1)
    alpha = 0
    beta = 0
    p: Union[np.ndarray, List[float]]
    for idx, p in enumerate(path):
        # first index is skipped
        if idx == 0:
            continue
        # stretch check
        if p[0] == path[idx - 1][0]:
            beta += 1
        else:
            # end beta count, add lambdas
            if beta > 0:
                beta += 1
                lmbd[idx - beta - 1 : idx - 1] = 1.0 - 1.0 / beta
                beta = 0
        # compression check
        if p[1] == path[idx - 1][1]:
            alpha += 1
        else:
            if alpha > 0:
                alpha += 1
                lmbd[idx - alpha : idx - 1] = 1.0 - alpha
                alpha = 0

    # check final entry
    if beta > 0:
        beta += 1
        lmbd[idx - beta : idx] = 1.0 - 1.0 / beta
        beta = 0
    if alpha > 0:
        alpha += 1
        lmbd[idx - alpha + 1 : idx] = 1.0 - alpha
        alpha = 0

    distortion_score = np.sum(np.abs(lmbd))
    compress_count = int(np.count_nonzero(lmbd > 0))
    stretch_count = int(np.count_nonzero(lmbd < 0))
    return lmbd, distortion_score, compress_count, stretch_count
