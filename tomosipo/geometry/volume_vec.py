import numpy as np
from tomosipo.utils import slice_interval
from numbers import Integral
import tomosipo as ts
from tomosipo import vector_calc as vc
from .transform import Transform


def volume_vec(*, shape, pos=0, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
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

    RT = ts.geometry.random_transform()
    vg = ts.volume_vec(
        shape=np.random.uniform(2, 10, size=3).astype(int),
        pos=np.random.normal(size=3),
    )
    return RT * vg


class VolumeVectorGeometry(object):
    """Documentation for VolumeVectorGeometry"""

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

        self._shape = ts.utils.to_shape(shape)
        pos = ts.utils.to_pos(pos)

        pos, w, v, u = np.broadcast_arrays(*(vc.to_vec(x) for x in (pos, w, v, u)))
        self._pos, self._w, self._v, self._u = pos, w, v, u

        shapes = [x.shape for x in [pos, w, v, u]]

        if min(shapes) != max(shapes):
            raise ValueError(
                "Not all arguments pos, w, v, u are the same shape. " f"Got: {shapes}"
            )

    def __repr__(self):
        with ts.utils.print_options():
            return (
                f"ts.volume_vec(\n"
                f"    shape={self._shape},\n"
                f"    pos={repr(self.pos)},\n"
                f"    w={repr(self.w)},\n"
                f"    v={repr(self.v)},\n"
                f"    u={repr(self.u)},\n"
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

    def __getitem__(self, key):
        """Slice the volume geometry

        The first index indexes into the number of steps. The next
        three indices index into space.

        To get the lower left voxel of the volume in the first time
        step, execute:

        >>> ts.volume().to_vec()[0, 0, 0, 0]

        :param key:
        :returns:
        :rtype:

        """
        full_slice = slice(None, None, None)

        if isinstance(key, tuple) and len(key) > 4:
            raise ValueError(
                f"VolumeVectorGeometry supports indexing in 4 dimensions. Got {key}."
            )

        if isinstance(key, Integral) or isinstance(key, slice):
            key = (key, full_slice, full_slice, full_slice)
        while isinstance(key, tuple) and len(key) < 4:
            key = (*key, full_slice)

        if isinstance(key, tuple) and len(key) == 4:
            w0, w1, lenW, stepW = slice_interval(
                0, self.shape[0], self.shape[0], key[1]
            )
            v0, v1, lenV, stepV = slice_interval(
                0, self.shape[1], self.shape[1], key[2]
            )
            u0, u1, lenU, stepU = slice_interval(
                0, self.shape[2], self.shape[2], key[3]
            )
            #
            new_shape = (lenW, lenV, lenU)
            # Calculate new lower-left corner, top-right corner, and center.
            new_llc = self.lower_left_corner + w0 * self.w + v0 * self.v + u0 * self.u
            new_trc = self.lower_left_corner + w1 * self.w + v1 * self.v + u1 * self.u
            new_pos = (new_llc + new_trc) / 2
            new_pos = new_pos[key[0]]
            #
            new_w = stepW * self.w[key[0]]
            new_v = stepV * self.v[key[0]]
            new_u = stepU * self.u[key[0]]

            return VolumeVectorGeometry(new_shape, new_pos, new_w, new_v, new_u)

        # If key is a boolean array, we may index like this:
        return VolumeVectorGeometry(
            shape=self.shape,
            pos=self.pos[key],
            w=self.w[key],
            v=self.v[key],
            u=self.u[key],
        )

    def __len__(self):
        return self.num_steps

    def to_vec(self):
        """Returns a volume vector geometry

        :returns: self
        :rtype: VolumeVectorGeometry

        """
        return self

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

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
    def shape(self):
        return self._shape

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
    def corners(self):
        """Returns a vector with the corners of the volume

        :returns: np.array
            Array with shape (num_steps, 8, 3), describing the 8
            corners of volume orientation in (Z, Y, X)-coordinates.
        :rtype: np.array
        """
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
    def lower_left_corner(self):
        """Returns a vector with the positions of the lower-left corner the object

        :returns: np.array
            Array with shape (num_steps, 3), describing the position
            of the lower-left corner of each volume orientation in (Z, Y,
            X)-coordinates.
        :rtype: np.array

        """
        llc = self._pos.copy()
        for vec, num_voxels in zip([self._w, self._v, self._u], self.shape):
            llc -= vec * num_voxels / 2
        return llc

    ###########################################################################
    #                          Transformation methods                         #
    ###########################################################################
    def reshape(self, new_shape):
        """Change the number of voxels without changing volume size

        :param new_shape: int or (int, int, int)
            The new shape of the detector in pixels in (w, v, u) direction.
        :returns: `self`
        :rtype: VolumeVectorGeometry

        """
        new_shape = ts.utils.to_shape(new_shape)
        new_w = self.sizes[:, 0] / max(new_shape[0], 1)
        new_v = self.sizes[:, 1] / max(new_shape[1], 1)
        new_u = self.sizes[:, 2] / max(new_shape[2], 1)

        return VolumeVectorGeometry(
            new_shape,
            self.pos,
            new_w,
            new_v,
            new_u,
        )

    def __rmul__(self, other):
        """Applies a projective matrix transformation to geometry

        :param other: `np.array`
            A transformation matrix
        :returns: A transformed geometry
        :rtype: `VolumeVectorGeometry`

        """
        if isinstance(other, Transform):
            new_pos = other.transform_point(self._pos)
            new_w = other.transform_vec(self._w)
            new_v = other.transform_vec(self._v)
            new_u = other.transform_vec(self._u)

            return VolumeVectorGeometry(self.shape, new_pos, new_w, new_v, new_u)
