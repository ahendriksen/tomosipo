import tomosipo as ts
import numpy as np
import warnings
from tomosipo.utils import up_tuple
import tomosipo.vector_calc as vc


class Transform(object):
    """Documentation for Transform"""

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

    def __getitem__(self, i):
        if not (isinstance(i, slice) or isinstance(i, int)):
            raise TypeError(
                f"Transform only support one-dimensional indexing. Got: {i}"
            )
        return Transform(self.matrix[i])

    @property
    def num_steps(self):
        return self.matrix.shape[0]

    @property
    def inv(self):
        return Transform(vc.invert_transformation_matrix(self.matrix))

    def transform_vec(self, vec):
        """Transform one or multiple vectors

        :param vec: `np.array`

        The following shapes are allowed:
        - `(n_rows, 3)`
        - `(3,)`

        :returns: `np.array`
        :rtype:

        The shape of the array is `(n_rows, 3)`

        """
        hv = vc.to_homogeneous_vec(vec)
        return vc.to_vec(vc.matrix_transform(self.matrix, hv))

    def transform_point(self, points):
        """Transform one or multiple points

        :param points: `np.array`

        The following shapes are allowed:
        - `(n_rows, 3)`
        - `(3,)`

        :returns: `np.array`
        :rtype:

        The shape of the array is `(n_rows, 3)`

        """
        hp = vc.to_homogeneous_point(points)
        return vc.to_vec(vc.matrix_transform(self.matrix, hp))


def identity():
    """Create identity transform

    :returns:
    :rtype:

    """
    return Transform(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))


def translate(t):
    """Create a translation transform

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


def scale(s, pos=None):
    """Create a scaling transform

    The scaling transform scales the coordinate frame of the object by `s`
    around position `pos`.

    The parameter `s` is interpreted as a series of homogeneous
    coordinates. You may pass in both homogeneous or non-homogeneous
    coordinates. Also, you may pass in multiple rows for multiple
    timesteps. The following shapes are allowed:

    - scalar
    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]


    :param s:
        By how much to scale in each direction.
    :param pos:  (optional)
        If not `None`, scale around a custom position instead of the
        origin.
    :returns:
    :rtype:

    """
    if np.isscalar(s):
        s = ts.types.to_size3d(s)
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

    if pos is not None:
        T = ts.translate(pos)
        return T * Transform(S) * T.inv
    else:
        return Transform(S)


def rotate(*, pos, axis, angles=None, rad=None, deg=None, right_handed=True):
    """Create a rotation transform

    The transform rotates around `axis` through position `pos` by some `angles`.

    The parameters `pos` and `axis` are interpreted as
    homogeneous coordinates. You may pass in both homogeneous or
    non-homogeneous coordinates. Also, you may pass in multiple rows
    for multiple timesteps. The following shapes are allowed:

    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]

    :param pos: `np.array` or `scalar`
        The position of the axis of rotation.
    :param axis:
        The direction vector of the axis of rotation.
    :param angles: `float` or `np.array`
        The angle by which must be rotated in radians.
    :param rad:
        **DEPRECATED**
    :param deg:
        **DEPRECATED**
    :param right_handed:
        By default, the rotation performs a right-handed rotation (in
        the anti-clockwise direction). A left-handed rotation is
        performed when `right_handed=False`.

    :returns:
    :rtype: Transform

    """
    if np.isscalar(pos):
        pos = ts.types.to_pos(pos)
    pos = vc.to_homogeneous_point(pos)
    axis = vc.to_homogeneous_vec(axis)
    pos, axis = vc._broadcastv(pos, axis)

    axis = axis / vc.norm(axis)

    angles_defined = angles is not None
    rad_deg_defined = rad is not None or deg is not None
    if not angles_defined and not rad_deg_defined:
        raise ValueError("The `angles=` parameter is required.")
    elif angles_defined and not rad_deg_defined:
        theta = angles
        # Make theta of shape `(num_steps, 1)`
        theta = vc.to_scalar(theta)
    elif angles_defined and rad_deg_defined:
        raise TypeError(
            "The `angles` parameter is not compatible with the `rad` or `deg` parameter. "
            "The `rad` and `deg` parameters are deprecated. "
        )
    else:
        # case: angles is None and (rad is not None or deg is not None):
        warnings.warn(
            "The `rad` and `deg` parameters of `ts.rotate` are deprecated. Please use `angles` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
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


def reflect(*, pos, axis):
    """Create a reflection transform

    The transform reflects in the plane through `pos` with normal vector `axis`.

    The parameters `pos` and `axis` are interpreted as
    homogeneous coordinates. You may pass in both homogeneous or
    non-homogeneous coordinates. Also, you may pass in multiple rows
    for multiple timesteps. The following shapes are allowed:

    - `(n_rows, 3)` [non-homogeneous] or `(n_rows, 4)` [homogeneous]
    - `(3,)` [non-homogeneous] or `(4,)` [homogeneous]

    :param pos: `np.array` or `scalar`
        A position intersecting the plane of reflection.
    :param axis:
        A normal vector to the plane of reflection. Need not be unit-normal.

    :returns:
    :rtype:

    """
    if np.isscalar(pos):
        pos = ts.utils.to_pos(pos)
    pos = vc.to_homogeneous_point(pos)
    axis = vc.to_homogeneous_vec(axis)
    pos, axis = vc._broadcastv(pos, axis)
    axis = axis / vc.norm(axis)[:, None]

    # Create a householder matrix for reflection through the origin.
    # https://en.wikipedia.org/wiki/Householder_transformation
    R_origin = ts.concatenate(
        [Transform(identity().matrix - 2 * np.outer(a, a)) for a in axis]
    )

    T = ts.translate(pos)

    return T * R_origin * T.inv


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
    :param box: `VolumeVectorGeometry` (optional)
        Retrieve `pos, w, v, u` arguments from a box.
    :returns:
    :rtype:

    """
    # TODO: Rename box argument
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
    :param box: `VolumeVectorGeometry` (optional)
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
    R = ts.rotate(pos=pos, axis=axis, angles=angle)
    S = ts.scale(abs(s))

    return R * S * T
