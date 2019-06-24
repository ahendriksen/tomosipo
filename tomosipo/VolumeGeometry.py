import astra
from collections import Iterator, Iterable
import numpy as np
from numbers import Integral
import warnings
import itertools
from .utils import up_tuple, slice_interval
import tomosipo as ts


def is_volume_geometry(g):
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
        return (
            isinstance(other, VolumeGeometry)
            and self.extent == other.extent
            and self.shape == other.shape
        )

    def __abs__(self):
        return np.prod(self.size())

    def __sub__(self, other):
        vg = self.copy()
        vg.extent = tuple(
            (ls - lo, rs - ro)
            for ((ls, rs), (lo, ro)) in zip(self.extent, other.extent)
        )
        return vg

    def __round__(self, places):
        c = self.copy()
        c.extent = tuple((round(l, places), round(r, places)) for l, r in c.extent)
        return c

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

    def size_floor(self):
        """Converts the extent of the volume to integers.

        This function makes the volume smaller. Use `size_ceil` to
        make the volume larger.

        Does not affect the shape (voxels) of the volume. Use
        `reshape` to change the shape.

        :returns: a new volume with integral extent
        :rtype: VolumeGeometry

        """
        c = self.copy()
        c.extent = tuple((np.ceil(l), np.floor(r)) for l, r in c.extent)
        return c

    def size_ceil(self):
        """Converts the extent of the volume to integers.

        This function makes the volume larger. Use `size_floor` to
        make the volume smaller.

        Does not affect the shape (voxels) of the volume. Use
        `reshape` to change the shape.

        :returns: a new volume with integral extent
        :rtype: VolumeGeometry

        """
        c = self.copy()
        c.extent = tuple((np.floor(l), np.ceil(r)) for l, r in c.extent)
        return c

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

    def to_box(self):
        """Returns an oriented box representating the volume

        :returns:
        :rtype:

        """
        return ts.OrientedBox(
            self.size(), self.get_center(), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        )

    def transform(self, matrix):
        warnings.warn(
            "Converting VolumeGeometry to OrientedBox. "
            "Use `T(vg.to_box())` to inhibit this warning. ",
            stacklevel=2,
        )
        return self.to_box().transform(matrix)

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


def volume_from_projection_geometry(projection_geometry, inside=False):
    """Fit a VolumeGeometry inside or around acquisition photon beam.

    Assumptions:

    1) All detectors are parallel to the z-axis. Specifically, v,
       the vector from detector pixel (0,0) to (1,0), must be
       parallel to the z-axis.

    2) This is a cone beam geometry. The code has not yet been
       tested with parallel beam geometries.

    :param projection_geometry: ProjectionGeometry
        Any projection geometry. Currently, only cone beam geometries
        are tested and supported.
    :param inside:
        Determines whether the volume should fit inside the cone
        angle, or outside the cone angle. For a 'normal'
        reconstruction, `inside=False` is the right default.
    :returns:
        A VolumeGeometry that fits within (inside=True) the cone
        angle, or a cube that hugs the cone angle from the outside.
    :rtype: VolumeGeometry

    """
    # TODO: perhaps a third option (apart from inside and outside)
    # should be provided, where a cylinder fits exactly inside the
    # VolumeGeometry and inside the photon beam.

    pg = projection_geometry.to_vector()

    if pg.is_parallel():
        warnings.warn(
            "volume_from_projection_geometry has not been tested with parallel geometries."
        )

    # Create a volume with the lowerleft corner on the origin which
    # has size maxi-mini. It is determined by the maximal extent
    # of the detector positions. If the ellipse below describes
    # the detector positions, then the square describes the cube
    # that we are creating below.
    #                                               (maxi)
    # +-----------------------------------------------+
    # |          ----/    ( detector ) \----          |
    # |      ---/         (  curve   )      \---      |
    # |    -/                                   \-    |
    # |  -/                                       \-  |
    # | /                                           \ |
    # |/                                             \|
    # | -------------------------------------------   |
    # |(                                           )  |
    # | -------------------------------------------   |
    # |\                (source curve)               /|
    # | \                                           / |
    # |  -\                                       /-  |
    # |    -\                                   /-    |
    # |      ---\                           /---      |
    # |          ----\                 /----          |
    # +-----------------------------------------------+
    # (mini)

    # Gather detector information
    detector_width = pg.shape[1]
    detector_height = pg.shape[0]

    # Create a (_, 3) shaped array of the corners
    corners = pg.get_corners()
    corners = np.concatenate(corners, axis=0)
    source_pos = pg.get_source_positions()
    all_pos = np.concatenate([corners, source_pos], axis=0)

    mini = np.min(all_pos, axis=0)
    maxi = np.max(all_pos, axis=0)
    # Make X and Y size equal.
    maxi[1:] = np.max(maxi[1:])
    mini[1:] = np.min(mini[1:])
    max_size = maxi - mini

    # Create cube that hits all source positions and detector corners.
    vg0 = VolumeGeometry().to_origin().multiply(max_size).to_center()

    # Depending on whether you want an inside or outside fit, the
    # preliminary best fit is either the maximal cube or a
    # minimally sized cube.
    s0 = np.array(vg0.size())
    if inside:
        s_best = np.array([ts.epsilon] * 3, dtype=np.int)
    else:
        s_best = np.copy(s0)

    # First, optimize the size of the volume in the XY plane and
    # then in the Z direction. We need a base mask and unit vector
    # to represent the possible cubes. Furthermore, we need a high
    # and low mark.
    axial_basis = (
        np.array([1, 0, 0], dtype=np.int),  # base mask
        np.array([0, 1, 1], dtype=np.int),  # unit vector
        ts.epsilon,  # low
        np.min(s0[1:]),  # high
        (0, detector_width),  # detector size
        (0, 1),  # detector comparison direction
    )

    z_basis = (
        np.array([0, 1, 1], dtype=np.int),  # base mask
        np.array([1, 0, 0], dtype=np.int),  # unit vector
        ts.epsilon,  # low
        s0[0],  # high
        (detector_height, 0),  # detector size
        (1, 0),  # detector comparison direction
    )

    for (base, unit, low, high, detector_size, cmp_v) in [axial_basis, z_basis]:
        detector_size = np.array(detector_size)
        cmp_v = np.array(cmp_v)
        detector_max = np.sum(np.abs(detector_size * cmp_v) / 2)
        base = s_best * base

        while ts.epsilon < high - low:
            mid = (low + high) / 2
            s = base + mid * unit
            v = vg0.scale(s / s0)

            on_detector = True
            all_corners_off = True
            for p in v.get_corners():
                projections = pg.project_point(p)
                pdot = np.abs(np.sum(cmp_v * projections, axis=1))

                # If ray is parallel to detector plane, we get an
                # np.nan. Hitting means that the ray hits the
                # detector plane, but not necessarily within the
                # boundaries of the detector.
                parallel = np.isnan(pdot)
                hitting = np.logical_not(parallel)

                on_detector = np.logical_and(hitting, on_detector)
                on_detector = np.logical_and(pdot < detector_max, on_detector)

                all_corners_off = np.logical_or(parallel, all_corners_off)
                all_corners_off = np.logical_and(detector_max < pdot, all_corners_off)

            # on_detector is `True` if all corners under any angle
            # are projected on the detector.
            on_detector = np.all(on_detector)

            # all_corners_off is `True` if there is an angle under
            # which all corners fall outside the detector
            all_corners_off = np.any(all_corners_off)

            go_up = on_detector if inside else not all_corners_off

            # print(
            #     f"go{'up' if go_up else 'down'}: {s} | {all_corners_off} | {on_detector}"
            # )
            # gtmp = Geometry(c0.scale_around_center(s / s0).to_astra(), self.astra_proj_geom)

            if go_up:
                low = mid
            else:
                high = mid

            # Save best if going up and inside
            #           or going down and outside
            if go_up == inside:
                s_best = np.copy(s)

    result_cube = vg0.scale(s_best / s0)
    return result_cube
