import numpy as np
import warnings
from tomosipo.utils import up_tuple
import tomosipo as ts
import itertools
from tomosipo import vector_calc as vc
from .transform import Transform


def box(size, pos, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
    """Create a new oriented box

    An oriented box with multiple orientations and positions.

    The position describes the center of the box.

    :param size: `(scalar, scalar, scalar)` or `scalar`
        The size of the box as measured in basis elements w,
        v, u.
    :param pos: `scalar`, `np.array`
        A numpy array of dimension (num_steps, 3)
        describing the center of the box in world-coordinates
        `(Z, Y, X)`. You may also pass a 3-tuple or a scalar.
    :param w: `np.array` (optional)
        A numpy array of dimension (num_steps, 3)
        describing the `w` basis element in `(Z, Y, X)`
        coordinates. Default is `(1, 0, 0)`.
    :param v: `np.array` (optional)
        A numpy array of dimension (num_steps, 3)
        describing the `v` basis element in `(Z, Y, X)`
        coordinates. Default is `(0, 1, 0)`.
    :param u: `np.array` (optional)
        A numpy array of dimension (num_steps, 3)
        describing the `u` basis element in `(Z, Y, X)`
        coordinates. Default is `(0, 0, 1)`.
    :returns:
    :rtype:

    """
    return OrientedBox(size, pos, w, v, u)


def random_box():
    """Create a random box

    Useful for testing.
    :returns: a box with random size, position and orientation.
    :rtype: OrientedBox

    """
    size = np.random.uniform(0, 10, size=3)
    pos = np.random.normal(size=3)

    RT = ts.geometry.random_transform()
    box = ts.box(size, pos)
    return RT * box


class OrientedBox(object):
    """Documentation for OrientedBox

    """

    def __init__(self, size, pos, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
        """Create a new oriented box

        An oriented box with multiple orientations and positions.

        The position describes the center of the box.

        :param size: `(scalar, scalar, scalar)` or `scalar`
            The size of the box as measured in basis elements w,
            v, u.
        :param pos: `scalar`, `np.array`
            A numpy array of dimension (num_steps, 3)
            describing the center of the box in world-coordinates
            `(Z, Y, X)`. You may also pass a 3-tuple or a scalar.
        :param w: `np.array` (optional)
            A numpy array of dimension (num_steps, 3)
            describing the `w` basis element in `(Z, Y, X)`
            coordinates. Default is `(1, 0, 0)`.
        :param v: `np.array` (optional)
            A numpy array of dimension (num_steps, 3)
            describing the `v` basis element in `(Z, Y, X)`
            coordinates. Default is `(0, 1, 0)`.
        :param u: `np.array` (optional)
            A numpy array of dimension (num_steps, 3)
            describing the `u` basis element in `(Z, Y, X)`
            coordinates. Default is `(0, 0, 1)`.
        :returns:
        :rtype:

        """
        super(OrientedBox, self).__init__()

        self._size = up_tuple(size, 3)
        if np.isscalar(pos):
            pos = up_tuple(pos, 3)

        pos, w, v, u = np.broadcast_arrays(*(vc.to_vec(x) for x in (pos, w, v, u)))
        self._pos, self._w, self._v, self._u = pos, w, v, u

        shapes = [x.shape for x in [pos, w, v, u]]

        if min(shapes) != max(shapes):
            raise ValueError(
                "Not all arguments pos, w, v, u are the same shape. " f"Got: {shapes}"
            )

    def __repr__(self):
        return (
            f"OrientedBox(\n"
            f"    size={self._size},\n"
            f"    pos={self.pos},\n"
            f"    w={self.w},\n"
            f"    v={self.v},\n"
            f"    u={self.u},\n"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, OrientedBox):
            return False

        d_pos = self.pos - other.pos
        d_w = self._size[0] * self._w - other._size[0] * other._w
        d_v = self._size[1] * self._v - other._size[1] * other._v
        d_u = self._size[2] * self._u - other._size[2] * other._u

        return (
            np.all(abs(d_pos) < ts.epsilon)
            and np.all(abs(d_w) < ts.epsilon)
            and np.all(abs(d_v) < ts.epsilon)
            and np.all(abs(d_u) < ts.epsilon)
        )

    @property
    def corners(self):
        c = np.array(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        )
        c = c - 0.5

        size = self._size
        c_w = c[:, 0:1, None] * self._w * size[0]
        c_v = c[:, 1:2, None] * self._v * size[1]
        c_u = c[:, 2:3, None] * self._u * size[2]

        c = self.pos + c_w + c_v + c_u

        return c.swapaxes(0, 1)

    @property
    def pos(self):
        return np.copy(self._pos)

    @property
    def w(self):
        return np.copy(self._w)

    @property
    def v(self):
        return np.copy(self._v)

    @property
    def u(self):
        return np.copy(self._u)

    @property
    def abs_size(self):
        """Returns the absolute size of the object

        The `size` property is defined relative to the local
        coordinate frame of the object. Therefore we must multiply by
        the length of the `w, u, v` vectors to obtain the absolute
        size of the object.

        Because the vectors `w, u, v` may change over time, `abs_size`
        is a vector with shape `(num_steps, 3)`.

        :returns: a numpy array of shape `(num_steps, 3)` describing the size of the object.
        :rtype:

        """
        W = self._size[0] * vc.norm(self._w)
        V = self._size[1] * vc.norm(self._v)
        U = self._size[2] * vc.norm(self._u)

        return np.array((W, V, U)).T

    @property
    def rel_size(self):
        """Returns the size of the box relative to its coordinate frame

        The size of the oriented box is defined relative to the local
        coordinate frame of the object. This relative size is returned
        by this property.

        Since the relative size does not change over time, it is a 3-tuple.

        :returns: The relative size of the box relative to its coordinate frame
        :rtype: (scalar, scalar, scalar)

        """
        return tuple(self._size)

    @property
    def size(self):
        """Use abs_size or rel_size instead!

        The size of an oriented box can be defined in two ways:

        1) Relative to its local coordinate frame. This is implemented
           by the the `rel_size` property.

        2) In absolute terms. This is implemented by the `abs_size`
           property.

        Therefore, it is recommended (and enforced) that you use
        `abs_size` or `rel_size`.

        :raises: DeprecationWarning
        :returns:
        :rtype:

        """

        msg = """Use abs_size or rel_size instead!

        The size of an oriented box can be defined in two ways:

        1) Relative to its local coordinate frame. This is implemented
           by the the `rel_size` property.

        2) In absolute terms. This is implemented by the `abs_size`
           property.

        Therefore, it is recommended (and enforced) that you use
        `abs_size` or `rel_size`.
        """
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        raise NotImplementedError(msg)

    @property
    def num_steps(self):
        """The number of orientations and positions of this box

        An oriented box can have multiple positions and orientations,
        similar to how an astra projection geometry has multiple
        angles. This property describes how many such "steps" are
        described by this object.

        :returns: the number of steps
        :rtype: `int`

        """
        return len(self._pos)

    def __rmul__(self, other):
        if isinstance(other, Transform):
            matrix = other.matrix
            pos = vc.to_homogeneous_point(self._pos)
            w = vc.to_homogeneous_vec(self._w)
            v = vc.to_homogeneous_vec(self._v)
            u = vc.to_homogeneous_vec(self._u)

            new_pos = vc.to_vec(vc.matrix_transform(matrix, pos))
            new_w = vc.to_vec(vc.matrix_transform(matrix, w))
            new_v = vc.to_vec(vc.matrix_transform(matrix, v))
            new_u = vc.to_vec(vc.matrix_transform(matrix, u))

            return OrientedBox(self.rel_size, new_pos, new_w, new_v, new_u)
