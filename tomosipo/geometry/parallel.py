import astra
import warnings
import numpy as np
import tomosipo as ts
from tomosipo.utils import up_tuple, up_slice
from .base_projection import ProjectionGeometry
from .parallel_vec import ParallelVectorGeometry
from .transform import Transform


def parallel(*, angles=1, shape=1, size=None):
    """Create a parallel-beam geometry

    :param angles: `np.array` or integral value
        If integral value: the number of angles in the parallel-beam
        geometry. This describes a full arc (2 pi radians) with
        uniform placement and without the start and end point
        overlapping.

        If np.array: the values of the array are taken as
        projection angle (units are radians).

    :param shape: (`int`, `int`) or `int`
        The detector shape in pixels. If tuple, the order is
        (height, width). Else the pixel has the same number of
        pixels in the U and V direction.
    :param size: (float, float) or float
        The detector size. If a single float is provided, the
        detector is square with equal width and height.

        The order is (height, width), i.e. (v, u).

    :returns: a parallel-beam geometry
    :rtype: ParallelGeometry

    """
    return ParallelGeometry(angles, shape, size)


def random_parallel():
    angles = np.random.normal(size=20)
    size = np.random.uniform(10, 20, size=2)
    shape = np.random.uniform(10, 20, size=2).astype(int)
    return parallel(angles=angles, shape=shape, size=size)


class ParallelGeometry(ProjectionGeometry):
    """A parametrized parallel-beam geometry

    """

    def __init__(self, angles=1, shape=1, size=None):
        """Create a parallel-beam geometry

            :param angles: `np.array` or integral value
                If integral value: the number of angles in the parallel-beam
                geometry. This describes a full arc (2 pi radians) with
                uniform placement and without the start and end point
                overlapping.

                If np.array: the values of the array are taken as
                projection angle (units are radians).
            :param size: (float, float) or float
                The detector size. If a single float is provided, the
                detector is square with equal width and height.

                The order is (height, width), i.e. (v, u).

            :param shape: (`int`, `int`) or `int`
                The detector shape in pixels. If tuple, the order is
                (height, width). Else the pixel has the same number of
                pixels in the U and V direction.
            :returns: a parallel-beam geometry
            :rtype: ParallelGeometry
        """
        # Shape
        super(ParallelGeometry, self).__init__(shape=shape)

        # Angles
        self._angles_original = angles
        if np.isscalar(angles):
            # 180°; first and last angle are not parallel.
            angles = np.linspace(0, np.pi, angles, endpoint=False)
        else:
            angles = np.array(angles, ndmin=1, dtype=np.float64)

        if len(angles) == 0:
            raise ValueError(
                f"ParallelGeometry expects non-empty array of angles; got {self._angles_original}"
            )
        if angles.ndim > 1:
            raise ValueError(
                f"ParallelGeometry expects one-dimensional array of angles; got {self._angles_original}"
            )
        self._angles = angles

        # Size
        if size is None:
            size = shape
        self._size = ts.utils.to_size(size, dim=2)

        self._is_cone = False
        self._is_parallel = True
        self._is_vector = False

    def __repr__(self):
        with ts.utils.print_options():
            return (
                f"ts.parallel(\n"
                f"    angles={repr(self._angles_original)},\n"
                f"    shape={repr(self.det_shape)},\n"
                f"    size={repr(self._size)},\n"
                f")"
            )

    def __eq__(self, other):
        if not isinstance(other, ParallelGeometry):
            return False

        diff_angles = np.array(self._angles) - np.array(other._angles)
        diff_size = np.array(self._size) - np.array(other._size)

        return (
            self.det_shape == other.det_shape
            and np.all(abs(diff_angles) < ts.epsilon)
            and np.all(abs(diff_size) < ts.epsilon)
        )

    def __getitem__(self, key):
        """Slice geometry by angle

        Example: Obtain geometry containing every second angle:

        >>> ts.parallel()[0::2]

        Indexing on the detector plane is not supported, since it
        might move the detector center.

        :param key:
        :returns:
        :rtype:

        """
        if isinstance(key, tuple):
            raise ValueError(
                f"Expected 1 index to ParallelGeometry, got {len(key)}. "
                f"Indexing on the detector plane is not supported, "
                f"since it might move the detector center. "
                f"To prevent this error, use `pg.to_vec()[a:b, c:d, e:f]'. "
            )

        new_angles = self._angles[key]
        if np.isscalar(new_angles):
            new_angles = np.array([new_angles])

        return parallel(angles=new_angles, shape=self.det_shape, size=self._size)

    def to_astra(self):
        row_count, col_count = self.det_shape
        det_spacing = np.array(self._size) / np.array(self.det_shape)
        return {
            "type": "parallel3d",
            "DetectorSpacingX": det_spacing[1],
            "DetectorSpacingY": det_spacing[0],
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "ProjectionAngles": np.copy(self._angles),
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "parallel3d":
            raise ValueError(
                "ParallelGeometry.from_astra only supports 'parallel3d' type astra geometries."
            )

        # Shape in v, u order (height, width)
        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        det_spacing = (astra_pg["DetectorSpacingY"], astra_pg["DetectorSpacingX"])
        angles = astra_pg["ProjectionAngles"]
        size = np.array(det_spacing) * np.array(shape)

        return parallel(angles=np.copy(angles), shape=shape, size=size)

    def to_vec(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        return ParallelVectorGeometry.from_astra(astra.geom_2vec(self.to_astra()))

    def to_box(self):
        """Returns an oriented box representating the detector

        :returns: an oriented box representating the detector
        :rtype:  `VolumeVectorGeometry`

        """
        return self.to_vec().to_box()

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @ProjectionGeometry.num_angles.getter
    def num_angles(self):
        return len(self._angles)

    @ProjectionGeometry.angles.getter
    def angles(self):
        return np.copy(self._angles)

    @ProjectionGeometry.src_pos.getter
    def src_pos(self):
        raise NotImplementedError()

    @ProjectionGeometry.det_pos.getter
    def det_pos(self):
        return self.to_vec().det_pos

    @ProjectionGeometry.det_v.getter
    def det_v(self):
        return self.to_vec().det_v

    @ProjectionGeometry.det_u.getter
    def det_u(self):
        return self.to_vec().det_u

    @ProjectionGeometry.det_normal.getter
    def det_normal(self):
        return self.to_vec().det_normal

    @ProjectionGeometry.ray_dir.getter
    def ray_dir(self):
        return self.to_vec().ray_dir

    @ProjectionGeometry.det_size.getter
    def det_size(self):
        return self._size

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        return np.repeat([self._size], self.num_angles, axis=0)

    @ProjectionGeometry.corners.getter
    def corners(self):
        return self.to_vec().corners

    @ProjectionGeometry.lower_left_corner.getter
    def lower_left_corner(self):
        return self.to_vec().lower_left_corner

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################

    def rescale_det(self, scale):
        scaleV, scaleU = up_tuple(scale, 2)
        scaleV, scaleU = int(scaleV), int(scaleU)
        shape = (self.det_shape[0] // scaleV, self.det_shape[1] // scaleU)

        return parallel(angles=np.copy(self.angles), shape=shape, size=self._size)

    def reshape(self, new_shape):
        return parallel(angles=np.copy(self._angles), shape=new_shape, size=self._size)

    def project_point(self, point):
        return self.to_vec().project_point(point)

    def __rmul__(self, other):
        if isinstance(other, Transform):
            warnings.warn(
                "Converting parallel geometry to vector geometry. "
                "Use `T * pg.to_vec()` to inhibit this warning. ",
                stacklevel=2,
            )
            return other * self.to_vec()
