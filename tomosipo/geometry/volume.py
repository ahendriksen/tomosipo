import astra
import numpy as np
import warnings
import tomosipo as ts
from .transform import Transform
from .volume_vec import VolumeVectorGeometry


def is_volume(g):
    """Determine if a geometry is a volume geometry

    The object can be a fixed volume geometry or a vector volume
    geometry.

    :param g: a geometry object
    :returns: `True` if `g` is a volume geometry
    :rtype: bool

    """
    return isinstance(g, VolumeGeometry) or isinstance(g, VolumeVectorGeometry)


def volume(shape=(1, 1, 1), pos=None, size=None, extent=None):
    """Create an axis-aligned volume geometry

    A VolumeGeometry is an axis-aligned cuboid centered on `pos`.

    You may provide a combination of arguments to create a new volume geometry:
    - shape (size will equal shape)
    - shape and pos (size will equal shape)
    - shape and size (volume will be centered on the origin)
    - shape, pos, and size.
    - shape and extent

    :param shape: `int` or (`int`, `int`, `int`)
        Shape of the voxel grid underlying the volume.
    :param extent: `(scalar, scalar)` or `((scalar, scalar), (scalar, scalar), (scalar, scalar))`
        The minimal and maximal value of the volume in the Z, Y, X
        coordinate. If only one minimal and maximal value is provided,
        then it is applied to all coordinates.
    :param center: `scalar` or `(scalar, scalar, scalar)`
    :param size: `scalar` or `(scalar, scalar, scalar)`
    :returns: a volume geometry of the unit cube with shape as given.
    :rtype: VolumeGeometry

    """
    shape = ts.utils.to_shape(shape)

    pos_present = pos is not None
    size_present = size is not None
    extent_present = extent is not None

    if extent_present and pos_present:
        raise ValueError(
            "ts.volume does not accept both `extent` and `pos` arguments. "
        )
    if extent_present and size_present:
        raise ValueError(
            "ts.volume does not accept both `extent` and `size` arguments. "
        )

    if pos is None and size is None and extent is None:
        # shape only
        return VolumeGeometry(shape, pos=0, size=shape)
    elif size is None and extent is None:
        # shape and pos
        return VolumeGeometry(shape, pos=pos, size=shape)
    elif pos is None and extent is None:
        # shape and size
        return VolumeGeometry(shape, pos=0, size=size)
    elif extent is None:
        # shape, pos, and size
        return VolumeGeometry(shape, pos=pos, size=size)
    elif extent_present:
        pos, size = _extent_to_pos_size(extent)
        return VolumeGeometry(shape, pos, size)

    assert (
        False
    ), "Dead code path. Please report error. Perhaps you passed `shape=None`?"


def random_volume():
    """Generates a random volume geometry

    :returns: a random volume geometry
    :rtype: `VolumeGeometry`

    """
    center, size = 1 + abs(np.random.normal(size=(2, 3)))
    shape = np.random.uniform(2, 100, 3).astype(int)
    return volume(shape).translate(center).scale(size)


def _pos_size_to_extent(pos, size):
    pos = np.array(ts.utils.to_pos(pos))
    size = np.array(ts.utils.to_size(size))

    min_extent = pos - 0.5 * size
    max_extent = pos + 0.5 * size

    return tuple((l, r) for l, r in zip(min_extent, max_extent))


def _extent_to_pos_size(extent):
    size = ts.utils.to_size(tuple(r - l for l, r in extent))
    pos = ts.utils.to_pos(tuple((r + l) / 2 for l, r in extent))

    return (pos, size)


class VolumeGeometry:
    """VolumeGeometry

    A VolumeGeometry describes a 3D axis-aligned cuboid that is
    divided into voxels.

    The number of voxels in each dimension determine the object's
    `shape'. The voxel size is thus determined by the size of
    the object and its shape.

    A VolumeGeometry cannot move in time and cannot be arbitrarily
    oriented. To obtain a moving volume geometry, convert the object
    to a vector representation using `.to_vec()`.

    """

    def __init__(self, shape=(1, 1, 1), pos=0, size=None):
        """Create a new volume geometry

        A VolumeGeometry is an axis-aligned cuboid centered on
        `pos`.

        If not `size' is not given, then the voxel size defaults to 1
        in each dimension.

        VolumeGeometry is indexed(Z, Y, X) just like numpy. The
        conversion to and from an astra_vol_geom depends on this.

        :param shape: `(int, int, int)` or `int`
            The shape of the volume as measured in voxels.
        :param pos: `0.0`, or `(scalar, scalar, scalar)`
            The center of the object. To center on the origin, pass
            `0` (the default).
        :param size: `scalar` or `(scalar, scalar, scalar)`
            The size of the object (must be non-negative). If `size`
            is a sing value, the volume equally sized in each
            dimension.
        :returns: a new volume geometry
        :rtype: `VolumeGeometry`

        """
        shape = ts.utils.to_shape(shape)
        pos = ts.utils.to_pos(pos)

        if size is None:
            # Make geometry where voxel size equals (1, 1, 1)
            self._inner = ts.volume_vec(shape, pos)
        else:
            size = ts.utils.to_size(size)
            # voxel size
            vs = tuple(sz / sh for sz, sh in zip(size, shape))
            w = (vs[0], 0, 0)
            v = (0, vs[1], 0)
            u = (0, 0, vs[2])
            self._inner = ts.volume_vec(shape, pos, w, v, u)

        np.array(self._inner.pos[0])
        np.array(self._inner.size)

    def __repr__(self):
        return (
            f"VolumeGeometry(\n"
            f"    shape={self._inner.shape},\n"
            f"    pos={tuple(self.pos[0])},\n"
            f"    size={self.size},\n"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, VolumeGeometry):
            return False
        return self._inner == other._inner

    def __getitem__(self, key):
        """Slice the volume geometry

        The key may be up to three-dimensional. Both examples below
        yield the same geometry describing the axial central slice:

        >>> ts.volume(128)[64, :, :]
        >>> ts.volume(128)[64]

        :param key:
        :returns:
        :rtype:

        """
        if isinstance(key, tuple):
            new_inner = self._inner[(0, *key)]
        else:
            new_inner = self._inner[0, key]

        return VolumeGeometry(new_inner.shape, new_inner.pos[0], new_inner.size,)

    def __contains__(self, other):
        """Check if other volume is contained in current volume

        :param other: VolumeGeometry
            Another volumegeometry.
        :returns: True if other Volume is contained in this one.
        :rtype: Boolean

        """
        # Find the left and right boundary in each dimension
        return all(
            s[0] <= o[0] and o[1] <= s[1] for s, o in zip(self.extent, other.extent)
        )

    # def __abs__(self):
    #     return np.prod(self.size)

    # def __sub__(self, other):
    #     vg = self.copy()
    #     vg.extent = tuple(
    #         (ls - lo, rs - ro)
    #         for ((ls, rs), (lo, ro)) in zip(self.extent, other.extent)
    #     )
    #     return vg

    def to_astra(self):
        """Return an Astra volume geometry.

        :returns:
        :rtype:

        """
        # astra.create_vol_geom(Y, X, Z, minx, maxx, miny, maxy, minz, maxz):
        #
        # :returns: A 3D volume geometry of size :math:`Y \times X \times
        # Z`, windowed as :math:`minx \leq x \leq maxx` and :math:`miny
        # \leq y \leq maxy` and :math:`minz \leq z \leq maxz` .

        v = self.shape
        e = self.extent
        return astra.create_vol_geom(v[1], v[2], v[0], *e[2], *e[1], *e[0])

    def to_vec(self):
        """Returns a vector representation of the volume

        :returns:
        :rtype: VolumeVectorGeometry

        """
        return self._inner

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @property
    def num_steps(self):
        """The number of orientations and positions of this volume

        A VolumeGeometry always has only a single step.
        """
        return 1

    @property
    def pos(self):
        return self._inner.pos

    @property
    def w(self):
        return self._inner.w

    @property
    def v(self):
        return self._inner.v

    @property
    def u(self):
        return self._inner.u

    @property
    def shape(self):
        return self._inner.shape

    @property
    def sizes(self):
        """Returns the absolute sizes of the volume

        For consistency with vector geometries, `sizes` is an array
        with shape `(1, 3)`.

        :returns: a numpy array of shape `(1, 3)` describing the size of the object.
        :rtype:

        """
        return self._inner.sizes

    @property
    def size(self):
        """Returns the absolute size of the volume

        :returns: the size in each dimension of the volume
        :rtype: `(scalar, scalar, scalar)`

        """
        return self._inner.size

    @property
    def voxel_sizes(self):
        """The voxel sizes of the volume

        *Note*: For consistency with vector geometries, `voxel_sizes`
         is an array with shape `(1, 3)`.

        :returns: a numpy array of shape `(1, 3)` describing the voxel size of the volume.
        :rtype: `np.array`

        """
        return self._inner.voxel_sizes

    @property
    def voxel_size(self):
        """The voxel size of the volume

        :returns: the size in each dimension of the volume
        :rtype: `(scalar, scalar, scalar)`
        """
        return self._inner.voxel_size

    @property
    def extent(self):
        """The extent of the volume in each dimension

        :returns: `((min_z, max_z), (min_y, max_y), (min_x, max_x))`
        :rtype: `((scalar, scalar), (scalar, scalar), (scalar, scalar))`

        """
        return _pos_size_to_extent(self._inner.pos[0], self._inner.size)

    @property
    def corners(self):
        """Returns a vector with the corners of the volume

        For consistency with the volume vector geometry, the returned
        array has leading dimension `1`.

        :returns: np.array
            Array with shape (1, 8, 3), describing the 8
            corners of volume orientation in (Z, Y, X)-coordinates.
        :rtype: np.array

        """
        return self._inner.corners

    @property
    def lower_left_corner(self):
        """Returns a vector with the positions of the lower-left corner the object

        For consistency with the volume vector geometry, the returned
        array has leading dimension `1`.

        :returns: np.array
            Array with shape (1, 3), describing the position of the
            lower-left corner of the volume in (Z, Y, X)-coordinates.
        :rtype: np.array

        """
        return self._inner.lower_left_corner

    ###########################################################################
    #                          Transformation methods                         #
    ###########################################################################

    def with_voxel_size(self, voxel_size):
        """Returns a new volume with the specified voxel size

        When the voxel_size does not cleanly divide the size of the
        volume, a volume is returned that is
        - centered on the origin;
        - fits inside the original volume;

        :param voxel_size:
        :returns:
        :rtype:

        """
        voxel_size = ts.utils.to_size(voxel_size)
        new_shape = (np.array(self.size) / voxel_size).astype(np.int)

        return VolumeGeometry(new_shape, pos=self.pos[0], size=new_shape * voxel_size)

    def reshape(self, new_shape):
        """Reshape the VolumeGeometry

        :param new_shape: `int` or (`int`, `int`, `int`)
            The new shape that the volume must have
        :returns:
            A new volume with the required shape
        :rtype: VolumeGeometry

        """
        return VolumeGeometry(new_shape, pos=self.pos[0], size=self.size,)

    def translate(self, t):
        t = ts.utils.to_pos(t)

        new_pos = tuple(p + t for p, t in zip(self.pos[0], t))

        return VolumeGeometry(shape=self.shape, pos=new_pos, size=self.size,)

    def untranslate(self, ts):
        return self.translate(-np.array(ts))

    def scale(self, scale):
        """Scales the volume around its center.

        The position of the volume does not change. This
        transformation does not affect the shape (voxels) of the
        volume. Use `reshape` to change the shape.

        :param scale: tuple or np.float
            By how much to scale the volume.
        :returns:
        :rtype:

        """
        scale = ts.utils.to_size(scale)
        new_size = tuple(a * b for a, b in zip(scale, self.size))
        return VolumeGeometry(shape=self.shape, pos=self.pos[0], size=new_size)

    def multiply(self, scale):
        """Scales the volume including its position.

        Does not affect the shape (voxels) of the volume. Use
        `reshape` to change the shape.

        :param scale: tuple or np.float
            By how much to scale the volume.
        :returns:
        :rtype:

        """
        scale = ts.utils.to_size(scale)
        new_size = tuple(a * b for a, b in zip(scale, self.size))
        new_pos = tuple(a * b for a, b in zip(scale, self.pos[0]))
        return VolumeGeometry(shape=self.shape, pos=new_pos, size=new_size)

    def __rmul__(self, other):
        """Applies a projective matrix transformation to geometry

        If the transformation does not rotate, but only translates and
        scale, a VolumeGeometry is returned. Otherwise, a
        VolumeVectorGeometry is returned.

        :param other: `ts.geometry.Transform`
            A transformation matrix
        :returns: A transformed geometry
        :rtype: `VolumeGeometry` or `VolumeVectorGeometry`

        """
        if isinstance(other, Transform):
            # Check if it is possible to apply transformation and
            # remain a VolumeGeometry.
            if other.num_steps == 1:
                translation = other.matrix[0, :3, 3]
                # NOTE: scale must be non-negative, otherwise ts.scale
                # throws an error. This should only be a problem when
                # `T != other`. That is why we wrap in `abs` below.
                scale = abs(other.matrix[0].diagonal()[:3])
                T = ts.translate(translation) * ts.scale(scale)
                if T == other:
                    # implement scaling and translation ourselves
                    return self.multiply(scale).translate(translation)

            # Convert to vector geometry and apply transformation
            warnings.warn(
                "Converting VolumeGeometry to VolumeVectorGeometry. "
                "Use `T * vg.to_vec()` to inhibit this warning. ",
                stacklevel=2,
            )
            return other * self.to_vec()


def from_astra(astra_vol_geom):
    """Converts an ASTRA 3D volume geometry to a VolumeGeometry

    :param astra_vol_geom: `dict`
        A dictionary created by `astra.create_vol_geom` that describes
        a 3D volume geometry.
    :returns: a tomosipo volume geometry
    :rtype: VolumeGeometry

    """
    avg = astra_vol_geom
    WindowMinX = avg["option"]["WindowMinX"]
    WindowMaxX = avg["option"]["WindowMaxX"]
    WindowMinY = avg["option"]["WindowMinY"]
    WindowMaxY = avg["option"]["WindowMaxY"]
    WindowMinZ = avg["option"]["WindowMinZ"]
    WindowMaxZ = avg["option"]["WindowMaxZ"]

    voxZ = avg["GridSliceCount"]
    voxY = avg["GridRowCount"]
    voxX = avg["GridColCount"]

    shape = (voxZ, voxY, voxX)
    extent = (
        (WindowMinZ, WindowMaxZ),
        (WindowMinY, WindowMaxY),
        (WindowMinX, WindowMaxX),
    )
    pos, size = _extent_to_pos_size(extent)
    return VolumeGeometry(shape=shape, pos=pos, size=size)
