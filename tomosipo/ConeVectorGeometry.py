import warnings
import numpy as np
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.utils import up_tuple, up_slice, slice_interval
from numbers import Integral
from tomosipo.ProjectionGeometry import ProjectionGeometry


def cone_vec(shape, source_positions, detector_positions, detector_vs, detector_us):
    """Create a cone beam vector geometry

    :param shape: (`int`, `int`) or `int`
        The detector shape in pixels. If tuple, the order is
        (height, width). Else the pixel has the same number of
        pixels in the U and V direction.
    :param source_positions: np.array
        A numpy array of dimension (num_positions, 3) with the
        source positions in (Z, Y, X) order.
    :param detector_positions:
        A numpy array of dimension (num_positions, 3) with the
        detector center positions in (Z, Y, X) order.
    :param detector_vs:
        A numpy array of dimension (num_positions, 3) with the
        vector pointing from the detector (0, 0) to (1, 0) pixel
        (up).
    :param detector_us:
        A numpy array of dimension (num_positions, 3) with the
        vector pointing from the detector (0, 0) to (0, 1) pixel
        (sideways).
    :returns:
    :rtype:

    """
    return ConeVectorGeometry(
        shape, source_positions, detector_positions, detector_vs, detector_us
    )


def random_cone_vec():
    """Generates a random cone vector geometry

    :returns: a random cone vector geometry
    :rtype: `ConeVectorGeometry`

    """
    angles = np.random.normal(size=20)
    size = np.random.uniform(10, 20, size=2)
    shape = np.random.uniform(10, 20, size=2)
    detector_distance = np.random.uniform(0, 10)
    source_distance = np.random.uniform(0, 10)

    pg = ts.cone(angles, size, shape, detector_distance, source_distance)
    return pg.to_vector()


class ConeVectorGeometry(ProjectionGeometry):
    """Documentation for ConeVectorGeometry

    A class for representing cone vector geometries.
    """

    def __init__(
        self, shape, source_positions, detector_positions, detector_vs, detector_us
    ):
        """Create a cone beam vector geometry

        :param shape: (`int`, `int`) or `int`
            The detector shape in pixels. If tuple, the order is
            (height, width). Else the pixel has the same number of
            pixels in the U and V direction.
        :param source_positions: np.array
            A numpy array of dimension (num_positions, 3) with the
            source positions in (Z, Y, X) order.
        :param detector_positions:
            A numpy array of dimension (num_positions, 3) with the
            detector center positions in (Z, Y, X) order.
        :param detector_vs:
            A numpy array of dimension (num_positions, 3) with the
            vector pointing from the detector (0, 0) to (1, 0) pixel
            (up).
        :param detector_us:
            A numpy array of dimension (num_positions, 3) with the
            vector pointing from the detector (0, 0) to (0, 1) pixel
            (sideways).
        :returns:
        :rtype:

        """
        super(ConeVectorGeometry, self).__init__(shape=shape)

        src_pos, det_pos, det_v, det_u = (
            vc.to_vec(x)
            for x in (source_positions, detector_positions, detector_vs, detector_us)
        )
        src_pos, det_pos, det_v, det_u = np.broadcast_arrays(
            src_pos, det_pos, det_v, det_u
        )

        vc.check_same_shapes(src_pos, det_pos, det_v, det_u)

        self.source_positions = src_pos
        self.detector_positions = det_pos
        self.detector_vs = det_v
        self.detector_us = det_u

        self._is_cone = True
        self._is_parallel = False
        self._is_vector = True

    def __repr__(self):
        return (
            f"(ConeVectorGeometry\n"
            f"    shape={self.shape},\n"
            f"    source_positions={self.source_positions},\n"
            f"    detector_positions={self.detector_positions},\n"
            f"    detector_vs={self.detector_vs},\n"
            f"    detector_us={self.detector_us}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, ConeVectorGeometry):
            return False

        dpos_diff = self.detector_positions - other.detector_positions
        spos_diff = self.source_positions - other.source_positions
        us_diff = self.detector_us - other.detector_us
        vs_diff = self.detector_vs - other.detector_vs

        return (
            self.shape == other.shape
            and np.all(abs(dpos_diff) < ts.epsilon)
            and np.all(abs(spos_diff) < ts.epsilon)
            and np.all(abs(us_diff) < ts.epsilon)
            and np.all(abs(vs_diff) < ts.epsilon)
        )

    @property
    def lower_left_corner(self):
        return (
            self.detector_positions
            - (self.detector_vs * self.shape[0]) / 2
            - (self.detector_us * self.shape[1]) / 2
        )

    @property
    def top_right_corner(self):
        return (
            self.detector_positions
            + (self.detector_vs * self.shape[0]) / 2
            + (self.detector_us * self.shape[1]) / 2
        )

    def __getitem__(self, key):
        full_slice = slice(None, None, None)

        if isinstance(key, Integral) or isinstance(key, slice):
            key = (key, full_slice, full_slice)
        while isinstance(key, tuple) and len(key) < 3:
            key = (*key, full_slice)

        if isinstance(key, tuple) and len(key) == 3:
            v0, v1, lenV, stepV = slice_interval(
                0, self.shape[0], self.shape[0], key[1]
            )
            u0, u1, lenU, stepU = slice_interval(
                0, self.shape[1], self.shape[1], key[2]
            )
            # Calculate new lower-left corner, top-right corner, and center.
            new_llc = (
                self.lower_left_corner + v0 * self.detector_vs + u0 * self.detector_us
            )
            new_trc = (
                self.lower_left_corner + v1 * self.detector_vs + u1 * self.detector_us
            )
            new_center = (new_llc + new_trc) / 2

            new_shape = (lenV, lenU)
            new_src_pos = self.source_positions[up_slice(key[0])]
            new_det_pos = new_center[up_slice(key[0])]
            new_det_vs = self.detector_vs[up_slice(key[0])] * stepV
            new_det_us = self.detector_us[up_slice(key[0])] * stepU

            return cone_vec(new_shape, new_src_pos, new_det_pos, new_det_vs, new_det_us)

    def to_astra(self):
        row_count, col_count = self.shape
        vectors = np.concatenate(
            [
                self.source_positions[:, ::-1],
                self.detector_positions[:, ::-1],
                self.detector_us[:, ::-1],
                self.detector_vs[:, ::-1],
            ],
            axis=1,
        )

        return {
            "type": "cone_vec",
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "Vectors": vectors,
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "cone_vec":
            raise ValueError(
                "ConeVectorGeometry.from_astra only supports 'cone' type astra geometries."
            )

        vecs = astra_pg["Vectors"]
        # ray direction (parallel) / source_position (cone)
        sp = vecs[:, :3]
        # detector pos:
        dp = vecs[:, 3:6]
        # Detector u and v direction
        us = vecs[:, 6:9]
        vs = vecs[:, 9:12]

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return ConeVectorGeometry(
            shape, sp[:, ::-1], dp[:, ::-1], vs[:, ::-1], us[:, ::-1]
        )

    def to_vector(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        return self

    def rescale_detector(self, scale):
        """Rescale detector pixels

        Rescales detector pixels without changing the size of the detector.

        :param scale: `int` or `(int, int)`
            Indicates how many times to enlarge a detector pixel. Per
            convention, the first coordinate scales the pixels in the
            `v` coordinate, and the second coordinate scales the
            pixels in the `u` coordinate.
        :returns: a rescaled cone vector geometry
        :rtype: `ConeVectorGeometry`

        """
        scaleV, scaleU = up_tuple(scale, 2)
        scaleV, scaleU = int(scaleV), int(scaleU)

        shape = (self.shape[0] // scaleV, self.shape[1] // scaleU)
        det_v = self.detector_vs / scaleV
        det_u = self.detector_us / scaleU

        return cone_vec(
            shape, self.source_positions, self.detector_positions, det_v, det_u
        )

    def project_point(self, point):
        """Projects point onto detectors

        This function projects onto the virtual detectors described by
        the projection geometry.

        For each source-detector pair, this function returns

            (detector_intersection_y, detector_intersection_x)

        where the distance is from the detector origin (center) and
        the units are in detector pixels.

        This function returns `np.nan` if the source-point ray is
        parallel to the detector plane.

        N.B:
        This function projects onto the detector even if the ray is
        moving away from the detector instead of towards it, or if the
        detector is between the source and point instead of behind the
        point.

        :param point: A three-dimensional vector (preferably np.array)
        :returns: np.array([[detector_intersection_y, detector_intersection_x],
                            .....])
        :rtype: np.array (num_angles * 2)

        """
        v_origin = np.array(point)

        # TODO: Check v_origin shape (should be 3)

        det_us = self.detector_us
        det_vs = self.detector_vs

        # (source_pos, det_o, det_y, det_x) = self._get_vectors()
        det_normal = vc.cross_product(det_us, det_vs)

        v_direction = v_origin - self.source_positions

        intersection = vc.intersect(
            v_origin, v_direction, self.detector_positions, det_normal
        )

        det_i_u = np.sum(intersection * det_us, axis=1) / vc.squared_norm(det_us)
        det_i_v = np.sum(intersection * det_vs, axis=1) / vc.squared_norm(det_vs)

        return np.stack((det_i_v, det_i_u), axis=-1)

    @ProjectionGeometry.detector_sizes.getter
    def detector_sizes(self):
        height = vc.norm(self.detector_vs * self.shape[0])
        width = vc.norm(self.detector_us * self.shape[1])

        return np.stack([height, width], axis=1)

    def transform(self, matrix):
        src_pos = vc.to_homogeneous_point(self.source_positions)
        det_pos = vc.to_homogeneous_point(self.detector_positions)
        det_v = vc.to_homogeneous_vec(self.detector_vs)
        det_u = vc.to_homogeneous_vec(self.detector_us)

        src_pos = vc.to_vec(vc.matrix_transform(matrix, src_pos))
        det_pos = vc.to_vec(vc.matrix_transform(matrix, det_pos))
        det_v = vc.to_vec(vc.matrix_transform(matrix, det_v))
        det_u = vc.to_vec(vc.matrix_transform(matrix, det_u))

        return ConeVectorGeometry(self.shape, src_pos, det_pos, det_v, det_u)

    def get_corners(self):
        """Returns a vector with the corners of each detector

        :returns: np.array
            Array with shape (4, num_angles, 3), describing the 4
            corners of each detector.
        :rtype: np.array

        """

        ds = self.detector_positions
        u_offset = self.detector_us * self.shape[1] / 2
        v_offset = self.detector_vs * self.shape[0] / 2

        return np.array(
            [
                ds - u_offset - v_offset,
                ds - u_offset + v_offset,
                ds + u_offset - v_offset,
                ds + u_offset + v_offset,
            ]
        )

    def get_source_positions(self):
        """Returns the vector with source positions

        :returns: np.array
            An array with shape (num_angles, 3) with the position of
            the source in (Z,Y,X) coordinates.
        :rtype: np.array

        """
        return np.copy(self.source_positions)

    def get_num_angles(self):
        """Return the number of angles in the projection geometry

        :returns:
            The number of angles in the projection geometry.
        :rtype: integer

        """
        return len(self.detector_positions)

    def to_box(self):
        """Returns two boxes representating the source and detector respectively

        :returns: (source_box, detector_box)
        :rtype:  `(OrientedBox, OrientedBox)`

        """
        src_pos = self.source_positions
        det_pos = self.detector_positions
        w = self.detector_vs  # v points up, w points up
        u = self.detector_us  # detector_u and u point in the same direction

        # TODO: Fix vc.norm so we do not need [:, None]
        # We do not want to introduce scaling, so we normalize w and u.
        w = w / vc.norm(w)[:, None]
        u = u / vc.norm(u)[:, None]
        # This is the detector normal and has norm 1. In right-handed
        # coordinates, it would point towards the source usually. Now
        # it points "into" the detector.
        v = vc.cross_product(u, w)

        # TODO: Warn when detector size changes during rotation.
        det_height, det_width = self.detector_sizes[0]

        if np.any(abs(np.ptp(self.detector_sizes, axis=0)) > ts.epsilon):
            warnings.warn(
                "The detector size is not uniform. "
                "Using first detector size for the box"
            )

        detector_box = ts.OrientedBox((det_height, 0, det_width), det_pos, w, v, u)

        # The source of course does not really have a size, but we
        # want to visualize it for now :)
        source_size = (det_width / 10,) * 3
        # We set the orientation of the source to be identical to
        # that of the detector.
        source_box = ts.OrientedBox(source_size, src_pos, w, v, u)

        return source_box, detector_box
