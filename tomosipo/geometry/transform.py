import tomosipo as ts
import numpy as np
from tomosipo.utils import up_tuple
import tomosipo.vector_calc as vc


class Transform(object):
    """Documentation for Transform

    """

    def __init__(self, matrix):
        super(Transform, self).__init__()
        self.matrix, _ = vc._broadcastmm(matrix, matrix)

    def __mul__(self, other):
        if isinstance(other, Transform):
            M = vc.matrix_matrix_transform(self.matrix, other.matrix)
            return Transform(M)
        else:
            return NotImplemented

    def __repr__(self):
        return f"Transform(\n" f"    {self.matrix}\n" f")"

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        A, B = vc._broadcastmm(self.matrix, other.matrix)

        return np.all(abs(A - B) < ts.epsilon)

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

    The parameter `t` is interpreted as a series homogeneous
    coordinates. You may pass in both homogeneous or non-homogeneous
    coordinates. Also, you may pass in multiple rows for multiple
    timesteps. The following shapes are allowed:

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
    """Return scaling transform

    The scaling transform scales the coordinate frame of the object.

    The parameter `s` is interpreted as a series of homogeneous
    coordinates. You may pass in both homogeneous or non-homogeneous
    coordinates. Also, you may pass in multiple rows for multiple
    timesteps. The following shapes are allowed:

    - scalar
    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]


    :param s:
        By how much to scale in each direction.
    :returns:
    :rtype:

    """
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



def rotate(pos, axis, *, rad=None, deg=None, right_handed=True):
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

    :param pos: `np.array` or `scalar`
        The position through which the axis moves.
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
    if np.isscalar(pos):
        pos = up_tuple(pos, 3)
    pos = vc.to_homogeneous_point(pos)
    axis = vc.to_homogeneous_vec(axis)
    pos, axis = vc._broadcastv(pos, axis)

    axis = axis / vc.norm(axis)

    if rad is None and deg is None:
        raise ValueError("At least one of `rad=` or `deg=` parameters is required.")
    # Define theta in radians
    theta = np.deg2rad(deg) if deg is not None else rad
    # Make theta of shape `(num_steps, 1)`
    theta = vc.to_scalar(theta)

    # Make the rotation left-handed if necessary
    if not right_handed:
        theta = -theta

    # Do the following:
    # 1) Translate such that `pos` is at the origin;
    # 2) Rotate around `axis`;
    # 3) Undo the translation.

    # 1) Translation matrix
    T = ts.translate(-pos)
    # 2) Rotation matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    axis_0, axis_1, axis_2 = axis[:, 0], axis[:, 1], axis[:, 2]
    R00 = np.cos(theta) + np.square(axis_0) * (1 - np.cos(theta))
    R01 = axis_0 * axis_1 * (1 - np.cos(theta)) - axis_2 * np.sin(theta)
    R02 = axis_0 * axis_2 * (1 - np.cos(theta)) + axis_1 * np.sin(theta)

    R10 = axis_1 * axis_0 * (1 - np.cos(theta)) + axis_2 * np.sin(theta)
    R11 = np.cos(theta) + np.square(axis_1) * (1 - np.cos(theta))
    R12 = axis_1 * axis_2 * (1 - np.cos(theta)) - axis_0 * np.sin(theta)

    R20 = axis_2 * axis_0 * (1 - np.cos(theta)) - axis_1 * np.sin(theta)
    R21 = axis_2 * axis_1 * (1 - np.cos(theta)) + axis_0 * np.sin(theta)
    R22 = np.cos(theta) + np.square(axis_2) * (1 - np.cos(theta))

    one = np.ones_like(R00)
    zero = np.zeros_like(R00)
    R0 = np.stack([R00, R01, R02, zero], axis=1)
    R1 = np.stack([R10, R11, R12, zero], axis=1)
    R2 = np.stack([R20, R21, R22, zero], axis=1)
    R3 = np.stack([zero, zero, zero, one], axis=1)
    R = Transform(np.concatenate([R0, R1, R2, R3], axis=2))

    return T.inv * R * T


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
    :param box: `OrientedBox` (optional)
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
    :param box: `OrientedBox` (optional)
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

    return R * S * T
