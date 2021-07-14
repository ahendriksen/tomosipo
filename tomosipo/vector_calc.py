"""This module provides functionality for vector calculus.

Many objects in tomosipo contain vectors that change over time. Think
of a `ts.ConeVectorGeometry`, for wich the source and detector
positions change with each angle. The same is true for a
`VolumeVectorGeometry`.

Therefore, it makes to sense to force these vectors into a common
format and provide some basic common functionality. In tomosipo, the
standard shape for these vector is `(num_steps, 3)`. The functionality
for creating and manipulating these vector can be found here.

This module provides a convenience function, `to_vec`, that converts
any vector that it deems compatible into the common vector format. For
scalars that change over time, such angles in a projection geometry or
angles in a rotation, it provides `to_scalar`, which converts a scalar
or array of scalars to a numpy array with shape `(num_steps, 1)`.

"""
import numpy as np
import tomosipo as ts
from contextlib import contextmanager


###############################################################################
#                               Vector Creation                               #
###############################################################################


def to_vec(x):
    x = np.array(x, dtype=np.float64, copy=False)
    s = x.shape
    x = np.array(x, dtype=np.float64, copy=False, ndmin=2)

    if x.ndim == 2 and x.shape[1] == 3:
        return x
    if x.ndim == 2 and x.shape[1] == 4:
        # Convert homogeneous vectors
        denom = np.copy(x[:, 3])[:, None]
        denom[abs(denom) < ts.epsilon] = 1.0
        return x[:, :3] / denom

    raise ValueError(f"Shape {s} cannot be converted to vector. ")


def to_scalar(x):
    x = np.array(x, dtype=np.float64, copy=False)
    s = x.shape
    x = np.array(x, dtype=np.float64, copy=False, ndmin=1)

    if x.ndim == 1:
        # Add a single dimension to the right. Note that `to_vec` adds
        # empty dimensions to the left.
        x = x[:, None]

    if x.ndim == 2 and x.shape[1] == 1:
        return x

    raise ValueError(f"Shape {s} cannot be converted to scalar. ")


def to_homogeneous(x, s):
    x, _ = _broadcastv(x, x)
    if x.ndim == 2:
        if x.shape[1] == 4:
            return x
        if x.shape[1] == 3:
            return np.append(x, s * np.ones((x.shape[0], 1)), axis=1)

    raise ValueError(
        "Could not convert array to homogeneous coordinates. "
        f"Expected shape (3,) or (n_rows, 3) but got {x.shape}"
    )


def to_homogeneous_vec(x):
    return to_homogeneous(x, 0)


def to_homogeneous_point(x):
    return to_homogeneous(x, 1)


###############################################################################
#                                 Broadcasting                                #
###############################################################################


def broadcast_lengths(len_a, len_b):
    if len_a == 1:
        return len_b
    if len_b == 1:
        return len_a
    if len_a == len_b:
        return len_a

    raise ValueError("Operands could not be broadcast together.")


def _broadcastv(x, y):
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)
    # Save original shapes for error messages later.
    s1, s2 = x.shape, y.shape
    # Add dimensions, if necessary:
    x = np.array(x, ndmin=2, copy=False)
    y = np.array(y, ndmin=2, copy=False)

    if x.shape[0] == 1:
        x = np.broadcast_to(x, (y.shape[0], x.shape[1]))
    elif y.shape[0] == 1:
        y = np.broadcast_to(y, (x.shape[0], y.shape[1]))

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(
            f"Arguments of shape {s1} and {s2} could not be broadcast together."
        )

    return x, y


def _broadcastmv(M, x):
    M, x = np.array(M, copy=False), np.array(x, copy=False)
    # Save original shapes for error messages later.
    s1, s2 = M.shape, x.shape

    M = np.array(M, ndmin=3, copy=False)
    x = np.array(x, ndmin=2, copy=False)

    if x.shape[0] == 1:
        x = np.broadcast_to(x, (M.shape[0], M.shape[2]))
    if M.shape[0] == 1:
        M = np.broadcast_to(M, (x.shape[0], M.shape[1], M.shape[2]))

    if (
        M.ndim != 3
        or x.ndim != 2
        or M.shape[2] != x.shape[1]
        or M.shape[0] != x.shape[0]
    ):
        raise ValueError(
            f"Arguments of shape {s1} and {s2} could not be broadcast together."
        )

    return M, x


def _broadcastmm(M1, M2):
    M1 = np.array(M1, copy=False)
    M2 = np.array(M2, copy=False)
    # Save original shapes for error messages
    s1 = M1.shape
    s2 = M2.shape
    M1 = np.array(M1, ndmin=3, copy=False)
    M2 = np.array(M2, ndmin=3, copy=False)

    if M1.shape[0] == 1:
        M1 = np.broadcast_to(M1, (M2.shape[0], M1.shape[1], M1.shape[2]))
    elif M2.shape[0] == 1:
        M2 = np.broadcast_to(M2, (M1.shape[0], M1.shape[1], M1.shape[2]))

    if (
        M1.ndim != 3
        or M2.ndim != 3
        or M1.shape[2] != M2.shape[1]
        or M1.shape[0] != M2.shape[0]
    ):
        raise ValueError(
            f"Arguments of shape {s1} and {s2} could not be broadcast together."
        )

    return M1, M2


###############################################################################
#                              Vector operations                              #
###############################################################################
def cross_product(x, y):
    x, y = _broadcastv(x, y)

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
    vo = to_vec(v_origin)
    vd = to_vec(v_direction)
    po = to_vec(plane_origin)
    pn = to_vec(plane_normal)

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


def dot(x, y):
    x, y = _broadcastv(x, y)
    return np.sum(x * y, axis=1)


def squared_norm(x):
    x, _ = _broadcastv(x, x)
    return dot(x, x)


def norm(x):
    return np.sqrt(squared_norm(x))


###############################################################################
#                          Coordinate transformations                         #
###############################################################################
def orthogonal_basis_from_axis(axis):
    axis0 = to_homogeneous_vec(axis)
    w0 = axis0[:, 0]
    w1 = axis0[:, 1]
    w2 = axis0[:, 2]
    zero = np.zeros_like(w0)
    one = np.ones_like(w0)

    with ignore_divide_by_zero():
        # When w0 == 0 and w1 == 0, use:

        # {a -> 0, b -> -f, c -> -Sqrt[1 - f^2],
        #  d -> 0, e -> -Sqrt[1 - f^2],
        #  g -> Sign[w2], h -> 0, i -> 0}
        a0 = zero
        b0 = -one
        c0 = zero
        d0 = zero
        e0 = zero
        f0 = (
            -one
        )  # Note that f is free (we use -1 instead of 1 to preserve left-handedness)
        g0 = w2 / np.abs(w2)
        h0 = zero
        i0 = zero

        # Otherwise, use:

        # {a -> w0/Sqrt[w0^2 + w1^2 + w2^2],
        #  b -> (w0 w2)/Sqrt[(w0^2 + w1^2) (w0^2 + w1^2 + w2^2)],
        #  c -> -(w1/Sqrt[w0^2 + w1^2]),
        #  d -> w1/Sqrt[w0^2 + w1^2 + w2^2],
        #  e -> (w1 w2)/Sqrt[(w0^2 + w1^2) (w0^2 + w1^2 + w2^2)],
        #  f -> w0/Sqrt[w0^2 + w1^2],
        #  g -> w2/Sqrt[w0^2 + w1^2 + w2^2],
        #  h -> -Sqrt[((w0^2 + w1^2)/(w0^2 + w1^2 + w2^2))]}
        a1 = w0 / norm(axis0)
        b1 = (w0 * w2) / np.sqrt((np.square(w0) + np.square(w1)) * squared_norm(axis0))
        c1 = -(w1 / np.sqrt(np.square(w0) + np.square(w1)))
        d1 = w1 / norm(axis0)
        e1 = (w1 * w2) / np.sqrt((np.square(w0) + np.square(w1)) * squared_norm(axis0))
        f1 = w0 / np.sqrt(np.square(w0) + np.square(w1))
        g1 = w2 / norm(axis0)
        h1 = -np.sqrt(((np.square(w0) + np.square(w1)) / squared_norm(axis0)))
        i1 = zero

    w0_zero = abs(w0) < ts.epsilon
    w1_zero = abs(w1) < ts.epsilon

    use0 = np.logical_and(w0_zero, w1_zero)
    use1 = np.logical_not(use0)

    a = np.nan_to_num(use0 * a0) + np.nan_to_num(use1 * a1)
    b = np.nan_to_num(use0 * b0) + np.nan_to_num(use1 * b1)
    c = np.nan_to_num(use0 * c0) + np.nan_to_num(use1 * c1)
    d = np.nan_to_num(use0 * d0) + np.nan_to_num(use1 * d1)
    e = np.nan_to_num(use0 * e0) + np.nan_to_num(use1 * e1)
    f = np.nan_to_num(use0 * f0) + np.nan_to_num(use1 * f1)
    g = np.nan_to_num(use0 * g0) + np.nan_to_num(use1 * g1)
    h = np.nan_to_num(use0 * h0) + np.nan_to_num(use1 * h1)
    i = np.nan_to_num(use0 * i0) + np.nan_to_num(use1 * i1)

    axis0 = np.stack([a, d, g], axis=1)
    axis1 = np.stack([b, e, h], axis=1)
    axis2 = np.stack([c, f, i], axis=1)

    return (axis0, axis1, axis2)


def matrix_transform(M, x):
    """Apply a projective matrix transformation to x

    :param M:
        A transformation matrix
    :param x: `np.array`
        A homogeneous coordinate vector
    :returns:
    :rtype:

    """

    M, x = _broadcastmv(M, x)

    assert M.ndim == 3
    assert x.shape[0] == M.shape[0]

    # The following code executes the equivalent of
    # > np.array([M_ @ x_ for M_, x_ in zip(M, x)])
    # (but substantially faster)
    return np.matmul(M, x[:, :, None])[:, :, 0]


def matrix_matrix_transform(M1, M2):
    M1, M2 = _broadcastmm(M1, M2)
    # The following code executes the equivalent of
    # > np.array([m1 @ m2 for m1, m2 in zip(M1, M2)])
    # (but substantially faster)
    return np.matmul(M1, M2)


def invert_transformation_matrix(M):
    M, _ = _broadcastmm(M, M)
    assert M.ndim == 3
    try:
        # The following code executes the equivalent of
        # > np.array([np.linalg.inv(m) for m in M])
        # (but substantially faster)
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise ValueError(f"Inverting matrix failed, {M}")


###############################################################################
#                              Utility functions                              #
###############################################################################
@contextmanager
def ignore_divide_by_zero():
    old_settings = np.seterr(divide="ignore", invalid="ignore")
    yield
    np.seterr(**old_settings)


def check_same_shapes(*args):
    shapes = [x.shape for x in args]
    if min(shapes) != max(shapes):
        raise ValueError(f"Not all arguments are the same shape. Got: {shapes}")
