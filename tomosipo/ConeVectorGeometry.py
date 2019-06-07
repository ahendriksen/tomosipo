import numpy as np
import astra
from .utils import up_tuple
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.ProjectionGeometry import ProjectionGeometry


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

        self.source_positions = np.array(source_positions)
        self.detector_positions = np.array(detector_positions)
        self.detector_vs = np.array(detector_vs)
        self.detector_us = np.array(detector_us)

        shapes = [
            self.source_positions.shape,
            self.detector_positions.shape,
            self.detector_vs.shape,
            self.detector_us.shape,
        ]

        s0 = shapes[0]

        if len(s0) != 2:
            raise ValueError("Source_positions must be two-dimensional.")

        if s0[1] != 3:
            raise ValueError("Source_positions must have 3 columns.")

        for s in shapes:
            if s != s0:
                raise ValueError(
                    "Not all shapes of source, detector, u, v vectors are equal"
                )

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

    def get_size(self):
        """Returns a vector with the size of each detector


        :returns: np.array
            Array with shape (num_angles, 2) in v and u direction
            (height x width)
        :rtype: np.array

        """
        height = vc.norm(self.detector_vs * self.shape[0])
        width = vc.norm(self.detector_us * self.shape[1])

        return np.stack([height, width], axis=1)

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
