import tomosipo as ts
import numpy as np
from .utils import up_tuple
import tomosipo.vector_calc as vc


class Transform(object):
    """Documentation for Transform

    """

    def __init__(self, matrix):
        super(Transform, self).__init__()
        self.matrix, _ = vc._broadcastmm(matrix, matrix)

    def __call__(self, x):
        if hasattr(x, "transform"):
            return x.transform(self.matrix)
        else:
            raise TypeError(
                f"Transform does not support transforming objects of type {type(x)}."
            )

    def __repr__(self):
        return f"Transform(\n" f"    {self.matrix}\n" f")"

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        A, B = vc._broadcastmm(self.matrix, other.matrix)

        return np.all(abs(A - B) < ts.epsilon)

    def transform(self, matrix):
        M = vc.matrix_matrix_transform(matrix, self.matrix)
        return Transform(M)

    @property
    def inv(self):
        return Transform(vc.invert_transformation_matrix(self.matrix))

    def norm(self):
        # XXX: Deprecate or is this useful?
        return vc.to_scalar([np.linalg.det(m) for m in self.matrix])


def identity():
    """Return identity transform

    :returns:
    :rtype:

    """
    return Transform(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))


def translate(t):
    """Return translation transform

    The parameter `t` is interpreted as a homogeneous coordinates. You
    may pass in both homogeneous or non-homogeneous coordinates. Also,
    you may pass in multiple rows for multiple timesteps. The
    following shapes are allowed:

    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]

    :param t: `np.array`
        By how much to translate.
    :returns: A transform describing the translation
    :rtype: `Transform`

    """
    t = vc.to_homogeneous_point(t)
    w = vc.to_homogeneous_vec((1, 0, 0))
    v = vc.to_homogeneous_vec((0, 1, 0))
    u = vc.to_homogeneous_vec((0, 0, 1))
    t, w, v, u = np.broadcast_arrays(t, w, v, u)
    return Transform(np.stack((w, v, u, t), axis=2))


def scale(s):
    if np.isscalar(s):
        s = up_tuple(s, 3)
    s = vc.to_vec(s)
    s0 = s[:, 0:1]  # scaling in coordinate 0
    s1 = s[:, 1:2]  # scaling in coordinate 1
    s2 = s[:, 2:3]  # scaling in coordinate 2

    zero = np.zeros_like(s0)
    one = np.ones_like(s0)
    S0 = np.stack([s0, zero, zero, zero], axis=1)
    S1 = np.stack([zero, s1, zero, zero], axis=1)
    S2 = np.stack([zero, zero, s2, zero], axis=1)
    S3 = np.stack([zero, zero, zero, one], axis=1)

    S = np.concatenate([S0, S1, S2, S3], axis=2)
    return Transform(S)


def rotate(position, axis, *, rad=None, deg=None, right_handed=True):
    """Rotate around axis through position by some angle

    The parameters `position` and `axis` are interpreted as
    homogeneous coordinates. You may pass in both homogeneous or
    non-homogeneous coordinates. Also, you may pass in multiple rows
    for multiple timesteps. The following shapes are allowed:

    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]

    *Note*: the (Z, Y, X) coordinate system is left-handed. If
    `right_handed=True` is set (the default), then the rotation is
    translated to a left-handed rotation matrix.

    :param position: `np.array`
        The position through which the axis moves
    :param axis:
        The axis of rotation.
    :param rad: `float` or `np.array`
        The angle by which must be rotated. Only one of `deg` or `rad` may be passed.
    :param deg: `float` or `np.array`
        The angle by which must be rotated. Only one of `deg` or `rad` may be passed.
    :param right_handed:

        By default, the rotation performs a right-handed rotation (in
        the anti-clockwise direction). If you want to perform a left-handed rotation, you may

    :returns:
    :rtype:

    """
    if rad is None and deg is None:
        raise ValueError("At least one of `rad=` or `deg=` parameters is required.")
    # Define theta in radians
    theta = np.deg2rad(deg) if deg is not None else rad
    # Make theta of shape `(num_steps, 1)`
    theta = vc.to_scalar(theta)

    # Make the rotation right-handed if necessary
    if right_handed:
        theta = -theta

    # Do the following:
    # 1) Move to a perspective where the axis of rotation aligns with the `Z` axis;
    # 2) Rotate around the `Z` axis;
    # 3) Undo the perspective change

    # Create perspective transformation:
    position = vc.to_homogeneous_point(position)
    axis = vc.to_homogeneous_vec(axis)
    a0, a1, a2 = vc.orthogonal_basis_from_axis(axis)
    S = from_perspective(position, a0, a1, a2)

    # Create rotation matrix
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    R0 = np.stack([one, zero, zero, zero], axis=1)
    R1 = np.stack([zero, np.cos(theta), np.sin(theta), zero], axis=1)
    R2 = np.stack([zero, -np.sin(theta), np.cos(theta), zero], axis=1)
    R3 = np.stack([zero, zero, zero, one], axis=1)
    R = Transform(np.concatenate([R0, R1, R2, R3], axis=2))

    # Do the following:
    # 1) Move to a perspective where the axis of rotation aligns with the `Z` axis;
    # 2) Rotate around the `Z` axis;
    # 3) Undo the perspective change
    #       (3) (2)(1)
    return S.inv(R)(S)


def to_perspective(pos=None, w=None, v=None, u=None, box=None):
    """Transform coordinate frame to another frame of reference

    Returns a coordinate transformation to another frame of reference
    (perspective).

    :param pos:
        The position of the new frame of reference.
    :param w:
        The first coordinate of the new frame of reference.
    :param v:
        The second coordinate of the new frame of reference.
    :param u:
        The second coordinate of the new frame of reference.
    :param box: `ts.OrientedBox` (optional)
        Retrieve `pos, w, v, u` arguments from a box.
    :returns:
    :rtype:

    """
    if box is not None:
        pos, w, v, u = box.pos, box.w, box.v, box.u

    if any(x is None for x in (pos, w, v, u)):
        raise ValueError(
            "Not enough arguments provided: one of pos, w, v, u is missing."
        )

    pos = vc.to_homogeneous_point(pos)
    w = vc.to_homogeneous_vec(w)
    v = vc.to_homogeneous_vec(v)
    u = vc.to_homogeneous_vec(u)
    pos, w, v, u = np.broadcast_arrays(pos, w, v, u)
    vc.check_same_shapes(pos, w, v, u)

    assert pos.ndim == 2
    return Transform(np.stack((w, v, u, pos), axis=2))


def from_perspective(pos=None, w=None, v=None, u=None, box=None):
    """Transform coordinate frame to another frame of reference

    Returns a coordinate transformation from another frame of
    reference (perspective) to the coordinate frame with origin (0, 0,
    0) and basis Z, Y, X.

    :param pos:
        The position of the new frame of reference.
    :param w:
        The first coordinate of the new frame of reference.
    :param v:
        The second coordinate of the new frame of reference.
    :param u:
        The second coordinate of the new frame of reference.
    :param box: `ts.OrientedBox` (optional)
        Retrieve `pos, w, v, u` arguments from a box.
    :returns:
    :rtype:

    """
    return to_perspective(pos, w, v, u, box).inv


def random_transform():
    """Generate a random transformation

    :returns: A random rigid transformation
    :rtype: `Transform`

    """
    t, pos, axis, s = np.random.normal(size=(4, 3))
    angle = np.random.normal()
    T = ts.translate(t)
    R = ts.rotate(pos, axis, rad=angle)
    S = ts.scale(s)

    return R(S)(T)
