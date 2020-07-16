import numpy as np
import warnings
from tomosipo.utils import up_tuple
import tomosipo as ts
from tomosipo import vector_calc as vc
from .transform import Transform


def volume_vec(shape, pos, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
    """Create a new volume vector geometry

    Like the parallel and cone vector geometries, the volume vector
    geometry can be arbitrarily oriented and positioned.

    The position describes the center of the volume.

    :param shape: `(int, int, int)` or `int`
        The shape of the volume as measured in basis elements w,
        v, u.
    :param pos: `scalar`, `np.array`
        A numpy array of dimension (num_steps, 3)
        describing the center of the volume in world-coordinates
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
    return VolumeVectorGeometry(shape, pos, w, v, u)


def random_volume_vec():
    """Create a random volume vector geometry

    Useful for testing.
    :returns: a volume vector geometry with random size, position and orientation.
    :rtype: VolumeVectorGeometry

    """
    shape = np.random.uniform(2, 10, size=3)
    pos = np.random.normal(size=3)

    RT = ts.geometry.random_transform()
    vg = ts.volume_vec(shape, pos)
    return RT * vg


class VolumeVectorGeometry(object):
    """Documentation for VolumeVectorGeometry

    """

    def __init__(self, shape, pos, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
        """Create a new volume vector geometry

        An arbitarily oriented volume with multiple orientations and positions.

        The position describes the center of the volume.

        :param shape: `(int, int, int)` or `int`
            The shape of the volume as measured in basis elements w,
            v, u.
        :param pos: `scalar`, `np.array`
            A numpy array of dimension (num_steps, 3)
            describing the center of the volume in world-coordinates
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
        super().__init__()

        self._shape = tuple(map(int, up_tuple(shape, 3)))
        if not np.all(s > 0 for s in self._shape):
            raise ValueError("Shape must be strictly positive.")

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
            f"VolumeVectorGeometry(\n"
            f"    shape={self._shape},\n"
            f"    pos={self.pos},\n"
            f"    w={self.w},\n"
            f"    v={self.v},\n"
            f"    u={self.u},\n"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, VolumeVectorGeometry):
            return False

        d_pos = self.pos - other.pos
        d_w = self._w - other._w
        d_v = self._v - other._v
        d_u = self._u - other._u

        return (
            self.shape == other.shape
            and np.all(abs(d_pos) < ts.epsilon)
            and np.all(abs(d_w) < ts.epsilon)
            and np.all(abs(d_v) < ts.epsilon)
            and np.all(abs(d_u) < ts.epsilon)
        )

    def __getitem__(self, i):
        # TODO: Implement subsampling the shape of the volume
        if not (isinstance(i, slice) or isinstance(i, int)):
            raise TypeError(
                f"VolumeVectorGeometry only support one-dimensional indexing. Got: {i}"
            )

        return VolumeVectorGeometry(
            self.shape, self.pos[i], self.w[i], self.v[i], self.u[i],
        )

    @property
    def shape(self):
        return self._shape

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

        shape = self._shape
        c_w = c[:, 0:1, None] * self._w * shape[0]
        c_v = c[:, 1:2, None] * self._v * shape[1]
        c_u = c[:, 2:3, None] * self._u * shape[2]

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
    def sizes(self):
        """Returns the absolute sizes of the volume

        The `size` property is defined relative to the local
        coordinate frame of the object.  Because the vectors `w, u, v`
        may change over time, `abs_size` is a vector with shape
        `(num_steps, 3)`.

        :returns: a numpy array of shape `(num_steps, 3)` describing the size of the object.
        :rtype:

        """
        W = self._shape[0] * vc.norm(self._w)
        V = self._shape[1] * vc.norm(self._v)
        U = self._shape[2] * vc.norm(self._u)

        return np.array((W, V, U)).T

    @property
    def size(self):
        """Returns the absolute size of the volume

        *Note*: Because the local coordinate frame of the volume may
         change over time, the size of the volume may also change. If
         this happens, this property throws a ValueError.

        :returns: the size in each dimension of the volume (if constant)
        :rtype: `(scalar, scalar, scalar)`

        """
        W, V, U = self.sizes.T

        w_constant = abs(min(W) - max(W)) < ts.epsilon
        v_constant = abs(min(V) - max(V)) < ts.epsilon
        u_constant = abs(min(U) - max(U)) < ts.epsilon

        if w_constant and v_constant and u_constant:
            return (W[0], V[0], U[0])
        else:
            raise ValueError(
                "The size of the volume is not constant. "
                "To prevent this error, use `vg.sizes'. "
            )

    @property
    def voxel_sizes(self):
        """The voxel sizes of the volume

        *Note*: Because the local coordinate frame of the volume may
         change over time, the voxel size of the volume may also change.

        :returns: a numpy array of shape `(num_steps, 3)` describing the voxel size of the volume.
        :rtype: `np.array`
        """
        return self.sizes / np.array([self.shape])

    @property
    def voxel_size(self):
        """The voxel size of the volume

        *Note*: Because the local coordinate frame of the volume may
         change over time, the voxel size of the volume may also
         change. If this happens, this property throws a ValueError.

        :returns: the size in each dimension of the volume (if constant)
        :rtype: `(scalar, scalar, scalar)`
        """
        return tuple(size / shape for size, shape in zip(self.size, self.shape))

    @property
    def num_steps(self):
        """The number of orientations and positions of this volume

        A volume vector geometry can have multiple positions and
        orientations, similar to how an astra projection geometry has
        multiple angles. This property describes how many such "steps"
        are described by this object.

        :returns: the number of steps
        :rtype: `int`

        """
        return len(self._pos)

    def to_vec(self):
        return self

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

            return VolumeVectorGeometry(self.shape, new_pos, new_w, new_v, new_u)
