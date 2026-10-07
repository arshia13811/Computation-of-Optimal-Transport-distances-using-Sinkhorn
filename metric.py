import numpy as np

METRICS = ("euclidean", "manhattan")


def make_ground_cost(metric, side=20, scale="raw"):
    if metric not in METRICS or scale not in ("raw", "median"):
        raise ValueError("Unknown metric or scaling rule.")
    if side < 2:
        raise ValueError("side must be at least 2.")
    coordinates = np.indices((side, side)).reshape(2, -1).T
    delta = np.abs(coordinates[:, None, :] - coordinates[None, :, :]).astype(float)
    if metric == "euclidean":
        M = np.sqrt(np.sum(delta ** 2, axis=2))
    elif metric == "manhattan":
        M = np.sum(delta, axis=2)
    else:
        M = np.max(delta, axis=2)
    if scale == "median":
        M = M / np.median(M)
    return M
