import astra
from collections import Iterator, Iterable
import numpy as np
from numbers import Integral
import warnings
import itertools
from tomosipo.utils import up_tuple, slice_interval
import tomosipo as ts
from .transform import Transform


def is_volume(g):
    """Determine if a geometry is a volume geometry

    A geometry object can be a volume geometry or a projection
    geometry.

    :param g: a geometry object
    :returns: `True` if `g` is a volume geometry
    :rtype: bool

    """
    return isinstance(g, VolumeGeometry)


def volume(shape=(1, 1, 1), extent=None, center=None, size=None):
    """Create a unit VolumeGeometry

    A VolumeGeometry is a unit cube centered on the origin. Each
    side has length 1.

    VolumeGeometry is indexed(Z, Y, X) just like numpy. The
    conversion to and from an astra_vol_geom depends on this.

    You may provide a combination of arguments to create a new volume geometry:
    - shape
    - shape and extent
    - shape and center and size

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
    vg = VolumeGeometry()
    shape = up_tuple(shape, 3)
    vg = vg.reshape(shape)

    if center is None and size is not None:
        raise ValueError("Both center and size must be provided.")
    if size is None and center is not None:
        raise ValueError("Both center and size must be provided.")
    if size is not None and center is not None and extent is not None:
        raise ValueError(
            "extent must not be provided when size and center already are."
        )

    if center is not None and size is not None:
        center = np.array(up_tuple(center, 3))
        size = np.array(up_tuple(size, 3))
        l = center - size / 2
        r = center + size / 2
        extent = tuple(zip(l, r))

    if extent is not None:
        if isinstance(extent, Iterator) or isinstance(extent, Iterable):
            if len(extent) == 2:
                extent = up_tuple((extent,), 3)
        try:
            (x, X), (y, Y), (z, Z) = extent
            for (a, A) in ((x, X), (y, Y), (z, Z)):
                if A < a:
                    raise ValueError(f"Extent (x, X) must satisfy x <= X; got: {(a,A)}")
            vg.extent = tuple(
                (float(a), float(A)) for (a, A) in ((x, X), (y, Y), (z, Z))
            )

        except TypeError:
            raise TypeError(f"Extent should be a tuple of ints, got {extent}")

    return vg


def random_volume():
    """Generates a random volume geometry

    :returns: a random volume geometry
    :rtype: `VolumeGeometry`

    """
    center, size = 1 + abs(np.random.normal(size=(2, 3)))
    shape = np.random.uniform(2, 100, 3)
    return volume(shape).translate(center).scale(size)


class VolumeGeometry:
    def __init__(self):
        """Create a unit VolumeGeometry

        A VolumeGeometry is a unit cube centered on the origin. Each
        side has length 1.

        VolumeGeometry is indexed(Z, Y, X) just like numpy. The
        conversion to and from an astra_vol_geom depends on this.

        :returns:
        :rtype:

        """
        self.extent = ((-0.5, 0.5),) * 3
        self.shape = (1, 1, 1)

    def __repr__(self):
        return f"VolumeGeometry < extent: {self.extent}, " f"shape: {self.shape}>"

    def __eq__(self, other):
        if not isinstance(other, VolumeGeometry):
            return False

        d_extent = np.array(self.extent) - np.array(other.extent)
        return self.shape == other.shape and np.all(abs(d_extent) < ts.epsilon)

    def __abs__(self):
        return np.prod(self.size())

    def __sub__(self, other):
        vg = self.copy()
        vg.extent = tuple(
            (ls - lo, rs - ro)
            for ((ls, rs), (lo, ro)) in zip(self.extent, other.extent)
        )
        return vg

    @property
    def voxel_size(self):
        return tuple(size / shape for size, shape in zip(self.size(), self.shape))

    @property
    def num_steps(self):
        """The number of orientations and positions of this volume

        A VolumeGeometry always has only a single step.
        """
        return 1

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
        voxel_size = up_tuple(voxel_size, 3)
        new_shape = (np.array(self.size()) / voxel_size).astype(np.int)

        return volume(new_shape, center=self.get_center(), size=new_shape * voxel_size)

    def reshape(self, new_shape):
        """Reshape the VolumeGeometry

        :param new_shape: `int` or (`int`, `int`, `int`)
            The new shape that the cube must have
        :returns:
            A new volume with the required shape
        :rtype: VolumeGeometry

        """
        new_shape = up_tuple(new_shape, 3)
        vg = self.copy()
        vg.shape = tuple(new_shape)

        return vg

    def size(self):
        """The size in real units of the volume.

        :returns: (Z-size, Y-size, X-size)
        :rtype: (np.int, np.int, np.int)

        """

        return tuple(r - l for l, r in self.extent)

    def displacement(self):
        """The distance from the origin of the volume.


        :returns: the coordinate of the  'lowerleft' corner of the volume.
        :rtype: (np.float, np.float, np.float)

        """

        return tuple(l for l, _ in self.extent)

    def get_center(self):
        return tuple(d + s / 2 for d, s in zip(self.displacement(), self.size()))

    def copy(self):
        vg = VolumeGeometry()
        vg.extent = self.extent
        vg.shape = self.shape
        return vg

    def to_center(self):
        """Center the volume on the origin.

        :returns:
        :rtype:

        """
        return self.untranslate(self.displacement()).untranslate(
            tuple(s / 2 for s in self.size())
        )

    def to_origin(self):
        """Move the lower-left corner of the volume to the origin.

        :returns: a moved volume.
        :rtype: VolumeGeometry

        """
        return self.untranslate(self.displacement())

    def translate(self, ts):
        ts = up_tuple(ts, 3)
        c = self.copy()
        c.extent = tuple((l + t, r + t) for (l, r), t in zip(c.extent, ts))
        return c

    def untranslate(self, ts):
        return self.translate(-np.array(ts))

    def multiply(self, scale):
        """Multiply the volume's extent by scale

        Does not affect the shape (voxels) of the volume. Use
        `reshape` to change the shape.

        :param scale: tuple or np.float
            By how much to multiply the extent of the volume.
        :returns: VolumeGeometry
        :rtype: VolumeGeometry

        """
        scale = up_tuple(scale, 3)
        c = self.copy()
        c.extent = tuple((l * t, r * t) for (l, r), t in zip(c.extent, scale))
        return c

    def scale(self, scale):
        """Scales the volume around its center.

        Does not affect the shape (voxels) of the volume. Use
        `reshape` to change the shape.

        :param scale: tuple or np.float
            By how much to scale the volume.
        :returns:
        :rtype:

        """

        center = self.get_center()
        return self.to_center().multiply(scale).translate(center)

    def __contains__(self, other):
        """Check if other volume is contained in current volume

        :param other: VolumeGeometry
            Another volumegeometry.
        :returns: True if other Volume is contained in this one.
        :rtype: Boolean

        """
        # TODO: Reverse order!! It looks like this function is the
        # wrong way around.

        # Find the left and right boundary in each dimension
        return all(
            s[0] <= o[0] and o[1] <= s[1] for s, o in zip(self.extent, other.extent)
        )

    def intersect(self, other):
        """Intersect this volume with another volume.

        The shape of the returned VolumeGeometry is undefined.

        :param other: VolumeGeometry
            Another volume geometry.
        :returns: None or a new VolumeGeometry
        :rtype: NoneType or VolumeGeometry

        """
        # Find the left and right boundary in each dimension
        ls = [max(s[0], o[0]) for s, o in zip(self.extent, other.extent)]
        rs = [min(s[1], o[1]) for s, o in zip(self.extent, other.extent)]
        # Is the left boundary smaller than the right boundary in each
        # dimension?
        separated = all([l < r for l, r in zip(ls, rs)])

        if separated:
            c = self.copy()
            c.extent = tuple(zip(ls, rs))
            return c
        else:
            return None

    def get_corners(self):
        """Compute corner coordinates of volume.

        :returns: (corner, corner, ... )
        :rtype: list of 3-tuples

        """
        return list(itertools.product(*self.extent))

    def to_vec(self):
        """Returns a vector representation of the volume

        :returns:
        :rtype: VolumeVectorGeometry

        """
        vs = self.voxel_size
        w = (vs[0], 0, 0)
        v = (0, vs[1], 0)
        u = (0, 0, vs[2])

        return ts.volume_vec(self.shape, self.get_center(), w, v, u)

    def __rmul__(self, other):
        if isinstance(other, Transform):
            # Check if it is possible to apply transformation and
            # remain a VolumeGeometry.
            if other.num_steps == 1:
                translation = other.matrix[0, :3, 3]
                scale = other.matrix[0].diagonal()[:3]
                T = ts.translate(translation) * ts.scale(scale)
                if T == other:
                    # implement scaling and translation ourselves
                    return self.scale(scale).translate(translation)

            # Convert to vector geometry and apply transformation
            warnings.warn(
                "Converting VolumeGeometry to VolumeVectorGeometry. "
                "Use `T * vg.to_vec()` to inhibit this warning. ",
                stacklevel=2,
            )
            return other * self.to_vec()

    def __getitem__(self, key):
        """Return self[key]

        :param key: An int, tuple of ints,
        :returns:
        :rtype:

        """
        full_slice = slice(None, None, None)
        one_key = isinstance(key, Integral) or isinstance(key, slice)
        if one_key:
            key = (key, full_slice, full_slice)

        while isinstance(key, tuple) and len(key) < 3:
            key = (*key, full_slice)

        if isinstance(key, tuple) and len(key) == 3:
            indices = [
                slice_interval(l, r, s, k)
                for ((l, r), s, k) in zip(self.extent, self.shape, key)
            ]

            new_shape = tuple(s for (_, _, s, _) in indices)
            new_extent = tuple((l, r) for (l, r, _, _) in indices)

            return volume(new_shape, new_extent)

        raise TypeError(
            f"Indexing a ConeGeometry with {type(key)} is not supported. "
            f"Try int or slice instead."
        )


def from_astra(avg):
    WindowMinX = avg["option"]["WindowMinX"]
    WindowMaxX = avg["option"]["WindowMaxX"]
    WindowMinY = avg["option"]["WindowMinY"]
    WindowMaxY = avg["option"]["WindowMaxY"]
    WindowMinZ = avg["option"]["WindowMinZ"]
    WindowMaxZ = avg["option"]["WindowMaxZ"]

    voxZ = avg["GridSliceCount"]
    voxY = avg["GridRowCount"]
    voxX = avg["GridColCount"]

    c = VolumeGeometry()
    c.extent = tuple(
        [(WindowMinZ, WindowMaxZ), (WindowMinY, WindowMaxY), (WindowMinX, WindowMaxX)]
    )

    c.shape = (voxZ, voxY, voxX)
    return c
