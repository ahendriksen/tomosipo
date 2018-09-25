"""
A module with utility functions for vector calculus.

All inputs and outputs are expected to be in a numpy array
shaped (n, 3) where n is some positive number.

"""
import numpy as np


def cross_product(x, y):
    N = np.stack(
        [
            x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
            x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
            x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0],
        ],
        axis=-1,
    )

    return N


def intersect(v_origin, v_direction, plane_origin, plane_normal):
    vo = v_origin
    vd = v_direction
    po = plane_origin
    pn = plane_normal

    np_err = np.geterr()
    try:
        np.seterr(divide="ignore")
        t = np.stack([np.sum(pn * (po - vo), axis=1) / np.sum(pn * vd, axis=1)], 1)
        # t[t < 0] = np.nan  # TODO: remove?
        t[np.isinf(t)] = np.nan
        intersection = vo + np.multiply(t, vd)
    finally:
        np.seterr(**np_err)

    return intersection


def squared_norm(x):
    return np.sum(x * x, axis=1)


def norm(x):
    return np.sqrt(squared_norm(x))
