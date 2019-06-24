import astra
import warnings
from numbers import Integral
import numpy as np
from .utils import up_tuple
import tomosipo as ts
from tomosipo.ProjectionGeometry import ProjectionGeometry
from tomosipo.ConeVectorGeometry import ConeVectorGeometry


def cone(angles=1, size=np.sqrt(2), shape=1, detector_distance=0, source_distance=2):
    """Create a cone beam geometry

        :param angles: `np.array` or integral value
            If integral value: the number of angles in the cone beam
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
        :param detector_distance: float
            The detector-origin distance.
        :param source_distance: float
            the source-origin distance.
        :returns: a cone beam geometry
        :rtype: ConeGeometry
    """
    return ConeGeometry(angles, size, shape, detector_distance, source_distance)


def random_cone():
    """Returns a random cone beam geometry

    *Note*: this geometry is not circular. It just samples random
     points on a circular path.

    :returns: a random cone geometry
    :rtype: `ConeGeometry`

    """
    angles = np.random.normal(size=20)
    size = np.random.uniform(10, 20, size=2)
    shape = np.random.uniform(10, 20, size=2)
    detector_distance = np.random.uniform(0, 10)
    source_distance = np.random.uniform(0, 10)

    return ts.cone(angles, size, shape, detector_distance, source_distance)


class ConeGeometry(ProjectionGeometry):
    """Documentation for ConeGeometry

    Cone beam geometry
    """

    def __getitem__(self, key):
        """Return self[key]

        :param key: An int, tuple of ints,
        :returns:
        :rtype:

        """

        one_key = isinstance(key, Integral) or isinstance(key, slice)
        if isinstance(key, Integral):
            key = slice(key, key + 1)
        if one_key:
            return ConeGeometry(
                angles=np.asarray(self.angles[key]),
                size=self.size,
                shape=self.shape,
                detector_distance=self.detector_distance,
                source_distance=self.source_distance,
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

    def __init__(
        self, angles=1, size=np.sqrt(2), shape=1, detector_distance=0, source_distance=2
    ):
        """Create a cone beam geometry

        :param angles: `np.array` or integral value
            If integral value: the number of angles in the cone beam
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
        :param detector_distance: float
            The detector-origin distance.
        :param source_distance: float
            the source-origin distance.
        :returns: a cone beam geometry
        :rtype: ConeGeometry

        """
        super(ConeGeometry, self).__init__(shape=shape)
        self.angles_original = angles
        if np.isscalar(angles):
            angles = np.linspace(0, 2 * np.pi, angles, False)
        else:
            angles = np.asarray(angles).astype(np.float64)

        if len(angles) == 0:
            raise ValueError(
                f"ConeGeometry expects non-empty array of angles; got {self.angles_original}"
            )
        size = up_tuple(size, 2)

        self.angles = angles
        self.size = size
        self.detector_distance = detector_distance
        self.source_distance = source_distance

        self._is_cone = True
        self._is_parallel = False
        self._is_vector = False

    def __repr__(self):
        # Use self.angles_original to make the representation fit on screen.
        return (
            f"ConeGeometry(\n"
            f"    angles={self.angles_original},\n"
            f"    size={self.size},\n"
            f"    shape={self.shape},\n"
            f"    detector_distance={self.detector_distance},\n"
            f"    source_distance={self.source_distance}\n"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, ConeGeometry):
            return False

        diff_angles = np.array(self.angles) - np.array(other.angles)
        diff_size = np.array(self.size) - np.array(other.size)
        diff_detector = self.detector_distance - other.detector_distance
        diff_source = self.source_distance - other.source_distance

        return (
            len(self.angles) == len(other.angles)
            and np.all(abs(diff_angles) < ts.epsilon)
            and np.all(abs(diff_size) < ts.epsilon)
            and np.all(self.shape == other.shape)
            and abs(diff_detector) < ts.epsilon
            and abs(diff_source) < ts.epsilon
        )

    def to_vector(self):
        return ConeVectorGeometry.from_astra(astra.geom_2vec(self.to_astra()))

    def to_astra(self):
        detector_spacing = np.array(self.size) / np.array(self.shape)
        return astra.create_proj_geom(
            "cone",
            detector_spacing[1],
            detector_spacing[0],  # u, then v (reversed)
            *self.shape,
            self.angles,
            self.source_distance,
            self.detector_distance,
        )

    def from_astra(astra_pg):
        pg_type = astra_pg["type"]
        if pg_type == "cone":
            det_spacing = (astra_pg["DetectorSpacingY"], astra_pg["DetectorSpacingX"])
            # Shape in v, u order (height, width)
            shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
            angles = astra_pg["ProjectionAngles"]
            size = np.array(det_spacing) * np.array(shape)
            return ConeGeometry(
                angles=angles,
                size=size,
                shape=shape,
                detector_distance=astra_pg["DistanceOriginDetector"],
                source_distance=astra_pg["DistanceOriginSource"],
            )
        else:
            raise ValueError(
                "ConeGeometry.from_astra only supports 'cone' type astra geometries."
            )

    def transform(self, matrix):
        warnings.warn(
            "Converting cone geometry to vector geometry. "
            "Use `T(pg.to_vec())` to inhibit this warning. ",
            stacklevel=2,
        )

        return self.to_vector().transform(matrix)

    @ProjectionGeometry.detector_sizes.getter
    def detector_sizes(self):
        return np.repeat([self.size], self.get_num_angles(), axis=0)

    def get_num_angles(self):
        """Return the number of angles in the projection geometry

        :returns:
            The number of angles in the projection geometry.
        :rtype: integer

        """
        return len(self.angles)

    def to_box(self):
        """Returns two boxes representating the source and detector respectively

        :returns: (source_box, detector_box)
        :rtype:  `(OrientedBox, OrientedBox)`

        """
        return self.to_vector().to_box()
