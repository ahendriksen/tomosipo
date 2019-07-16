import astra
import warnings
from numbers import Integral
import numpy as np
import tomosipo as ts
from tomosipo.utils import up_tuple, up_slice
from .base_projection import ProjectionGeometry
from .cone_vec import ConeVectorGeometry
from .transform import Transform


def cone(angles=1, size=np.sqrt(2), shape=1, src_obj_dist=2.0, src_det_dist=2.0):
    """Create a cone-beam geometry

    :param angles: `np.array` or integral value
        If integral value: the number of angles in the cone-beam
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
    :param src_obj_dist: scalar
        The source to object distance.
    :param src_det_dist:
        The source to detector distance.
    :returns: a cone-beam geometry
    :rtype: ConeGeometry
    """
    return ConeGeometry(angles, size, shape, src_obj_dist, src_det_dist)


def random_cone():
    """Returns a random cone-beam geometry

    *Note*: this geometry is not circular. It just samples random
     points on a circular path.

    :returns: a random cone geometry
    :rtype: `ConeGeometry`

    """
    angles = np.random.normal(size=20)
    size = np.random.uniform(10, 20, size=2)
    shape = np.random.uniform(10, 20, size=2)
    src_obj_dist = np.random.uniform(0, 10)
    obj_det_dist = np.random.uniform(0, 10)
    src_det_dist = src_obj_dist + obj_det_dist

    return ts.cone(
        angles, size, shape, src_obj_dist=src_obj_dist, src_det_dist=src_det_dist
    )


class ConeGeometry(ProjectionGeometry):
    """A parametrized circular cone-beam geometry
    """

    def __init__(
        self, angles=1, size=np.sqrt(2), shape=1, src_obj_dist=2.0, src_det_dist=2.0
    ):
        """Create a cone-beam geometry

        :param angles: `np.array` or integral value
            If integral value: the number of angles in the cone-beam
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
        :param src_obj_dist: scalar
            The source to object distance.
        :param src_det_dist:
            The source to detector distance.
        :returns: a cone-beam geometry
        :rtype: ConeGeometry
        """
        super(ConeGeometry, self).__init__(shape=shape)
        self.angles_original = angles
        if np.isscalar(angles):
            # XXX: Should maybe include endpoints? -----v
            angles = np.linspace(0, 2 * np.pi, angles, False)
        else:
            angles = np.array(angles, copy=False, ndmin=1, dtype=np.float64)

        if len(angles) == 0:
            raise ValueError(
                f"ConeGeometry expects non-empty array of angles; got {self.angles_original}"
            )
        size = up_tuple(size, 2)

        self._angles = angles
        self._size = tuple(size)
        self._src_obj_dist = float(src_obj_dist)
        self._src_det_dist = float(src_det_dist)

        self._is_cone = True
        self._is_parallel = False
        self._is_vector = False

    def __repr__(self):
        # Use self.angles_original to make the representation fit on screen.
        return (
            f"ConeGeometry(\n"
            f"    angles={self.angles_original},\n"
            f"    size={self.det_size},\n"
            f"    shape={self.det_shape},\n"
            f"    src_obj_dist={self._src_obj_dist},\n"
            f"    src_det_dist={self._src_det_dist}\n"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, ConeGeometry):
            return False

        diff_angles = np.array(self.angles) - np.array(other.angles)
        diff_size = np.array(self.det_size) - np.array(other.det_size)
        diff_sod = self.src_obj_dist - other.src_obj_dist
        diff_sdd = self.src_det_dist - other.src_det_dist

        return (
            len(self.angles) == len(other.angles)
            and np.all(abs(diff_angles) < ts.epsilon)
            and np.all(abs(diff_size) < ts.epsilon)
            and np.all(self.det_shape == other.det_shape)
            and abs(diff_sod) < ts.epsilon
            and abs(diff_sdd) < ts.epsilon
        )

    def __getitem__(self, key):
        """Return self[key]

        :param key: An int, tuple of ints,
        :returns:
        :rtype:

        """

        one_key = isinstance(key, Integral) or isinstance(key, slice)

        if one_key:
            key = up_slice(key)
            return ConeGeometry(
                angles=np.asarray(self.angles[key]),
                size=self.det_size,
                shape=self.det_shape,
                src_obj_dist=self.src_obj_dist,
                src_det_dist=self.src_det_dist,
            )
        if isinstance(key, tuple):
            raise IndexError(
                f"Expected 1 index to ConeGeometry, got {len(key)}. "
                f"Indexing on the detector plane is not supported, "
                f"since it might move the detector center. "
            )
        raise TypeError(
            f"Indexing a ConeGeometry with {type(key)} is not supported. "
            f"Try int or slice instead."
        )

    def to_vec(self):
        return ConeVectorGeometry.from_astra(astra.geom_2vec(self.to_astra()))

    def to_astra(self):
        detector_spacing = np.array(self.det_size) / np.array(self.det_shape)
        origin_det_dist = self.src_det_dist - self.src_obj_dist
        return astra.create_proj_geom(
            "cone",
            detector_spacing[1],
            detector_spacing[0],  # u, then v (reversed)
            *self.det_shape,
            self.angles,
            self.src_obj_dist,
            origin_det_dist,
        )

    def from_astra(astra_pg):
        pg_type = astra_pg["type"]
        if pg_type == "cone":
            det_spacing = (astra_pg["DetectorSpacingY"], astra_pg["DetectorSpacingX"])
            # Shape in v, u order (height, width)
            shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
            angles = astra_pg["ProjectionAngles"]
            size = np.array(det_spacing) * np.array(shape)

            origin_det_dist = astra_pg["DistanceOriginDetector"]
            src_origin_dist = astra_pg["DistanceOriginSource"]
            return ConeGeometry(
                angles=angles,
                size=size,
                shape=shape,
                src_obj_dist=src_origin_dist,
                src_det_dist=src_origin_dist + origin_det_dist,
            )
        else:
            raise ValueError(
                "ConeGeometry.from_astra only supports 'cone' type astra geometries."
            )

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @property
    def src_obj_dist(self):
        """The source to object distance
        """
        return self._src_obj_dist

    @property
    def src_det_dist(self):
        """The source to detector distance
        """
        return self._src_det_dist

    @ProjectionGeometry.num_angles.getter
    def num_angles(self):
        return len(self.angles)

    @ProjectionGeometry.angles.getter
    def angles(self):
        return np.copy(self._angles)

    @ProjectionGeometry.src_pos.getter
    def src_pos(self):
        return self.to_vec().src_pos

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
        raise NotImplementedError()

    @ProjectionGeometry.det_size.getter
    def det_size(self):
        return self._size

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        return np.repeat([self.det_size], self.num_angles, axis=0)

    @ProjectionGeometry.corners.getter
    def corners(self):
        return self.to_vec().corners

    @ProjectionGeometry.lower_left_corner.getter
    def lower_left_corner(self):
        return self.to_vec().lower_left_corner

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################
    def __rmul__(self, other):
        if isinstance(other, Transform):
            warnings.warn(
                "Converting cone geometry to vector geometry. "
                "Use `T * pg.to_vec()` to inhibit this warning. ",
                stacklevel=2,
            )

            return other * self.to_vec()

    def rescale_det(self, scale):
        scaleV, scaleU = up_tuple(scale, 2)
        scaleV, scaleU = int(scaleV), int(scaleU)

        shape = (self.det_shape[0] // scaleV, self.det_shape[1] // scaleU)

        return cone(
            self.angles_original,
            self.det_size,
            shape,
            self.src_obj_dist,
            self.src_det_dist,
        )

    def reshape(self, new_shape):
        new_shape = up_tuple(new_shape, 2)
        return cone(
            self.angles_original,
            self.det_size,
            new_shape,
            self.src_obj_dist,
            self.src_det_dist,
        )

    def project_point(self, point):
        return self.to_vec().project_point(point)

    def to_box(self):
        """Returns two boxes representating the source and detector respectively

        :returns: (source_box, detector_box)
        :rtype:  `(OrientedBox, OrientedBox)`

        """
        return self.to_vec().to_box()
