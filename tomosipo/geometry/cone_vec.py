import warnings
import numpy as np
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.utils import up_tuple, up_slice, slice_interval
from numbers import Integral
from .base_projection import ProjectionGeometry
from . import det_vec as dv


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
    return pg.to_vec()


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

        self._src_pos = src_pos
        self._det_vec = dv.det_vec(shape, det_pos, det_v, det_u)

        self._is_cone = True
        self._is_parallel = False
        self._is_vector = True

    def __repr__(self):
        return (
            f"(ConeVectorGeometry\n"
            f"    shape={self.det_shape},\n"
            f"    source_positions={self._src_pos},\n"
            f"    detector_positions={self._det_vec.det_pos},\n"
            f"    detector_vs={self._det_vec.det_v},\n"
            f"    detector_us={self._det_vec.det_u}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, ConeVectorGeometry):
            return False

        spos_diff = self._src_pos - other.src_pos

        return self._det_vec == other._det_vec and np.all(abs(spos_diff) < ts.epsilon)

    def __getitem__(self, key):
        det_vec = self._det_vec[key]
        if isinstance(key, tuple):
            key, *_ = key

        new_src_pos = self._src_pos[up_slice(key)]
        return cone_vec(
            det_vec.det_shape,
            new_src_pos,
            det_vec.det_pos,
            det_vec.det_v,
            det_vec.det_u,
        )

    def to_astra(self):
        row_count, col_count = self.det_shape
        vectors = np.concatenate(
            [
                self._src_pos[:, ::-1],
                self._det_vec.det_pos[:, ::-1],
                self._det_vec.det_u[:, ::-1],
                self._det_vec.det_v[:, ::-1],
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

    def to_vec(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        return self

    def to_box(self):
        """Returns two boxes representating the source and detector respectively

        :returns: (source_box, detector_box)
        :rtype:  `(OrientedBox, OrientedBox)`

        """

        det_box = self._det_vec.to_box()

        # The source of course does not really have a size, but we
        # want to be able to visualize it, so we take the height
        # divided by 10.
        source_size = (det_box.size[0] / 10,) * 3
        # We set the orientation of the source to be identical to
        # that of the detector.
        src_box = ts.box(source_size, self.src_pos, det_box.w, det_box.v, det_box.u)

        return src_box, det_box

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @ProjectionGeometry.num_angles.getter
    def num_angles(self):
        return self._det_vec.num_angles

    @ProjectionGeometry.src_pos.getter
    def src_pos(self):
        return np.copy(self._src_pos)

    @ProjectionGeometry.det_pos.getter
    def det_pos(self):
        return self._det_vec.det_pos

    @ProjectionGeometry.det_v.getter
    def det_v(self):
        return self._det_vec.det_v

    @ProjectionGeometry.det_u.getter
    def det_u(self):
        return self._det_vec.det_u

    # TODO: det_normal

    @ProjectionGeometry.ray_dir.getter
    def ray_dir(self):
        raise NotImplementedError()

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        return self._det_vec.det_sizes

    @ProjectionGeometry.corners.getter
    def corners(self):
        return self._det_vec.corners

    @property
    def lower_left_corner(self):
        return self._det_vec.lower_left_corner

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################

    def rescale_det(self, scale):
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

        det_vec = self._det_vec.rescale_det(scale)

        return cone_vec(
            det_vec.det_shape,
            self._src_pos,
            det_vec.det_pos,
            det_vec.det_v,
            det_vec.det_u,
        )

    def reshape(self, new_shape):
        """Reshape detector pixels without changing detector size


        :param new_shape: int or (int, int)
            The new shape of the detector in pixels in `v` (height)
            and `u` (width) direction.
        :returns: `self`
        :rtype: ProjectionGeometry

        """
        det_vec = self._det_vec.reshape(new_shape)
        return cone_vec(
            new_shape, self.src_pos, det_vec.det_pos, det_vec.det_v, det_vec.det_u
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
        if np.isscalar(point):
            point = up_tuple(point, 3)

        v_origin = vc.to_vec(point)

        # TODO: Check v_origin shape (should be 3)

        det_pos = self._det_vec.det_pos
        det_v = self._det_vec.det_v
        det_u = self._det_vec.det_u

        # (source_pos, det_o, det_y, det_x) = self._get_vectors()
        det_normal = vc.cross_product(det_u, det_v)

        v_direction = v_origin - self._src_pos

        intersection = vc.intersect(v_origin, v_direction, det_pos, det_normal)

        det_i_u = np.sum(intersection * det_u, axis=1) / vc.squared_norm(det_u)
        det_i_v = np.sum(intersection * det_v, axis=1) / vc.squared_norm(det_v)

        return np.stack((det_i_v, det_i_u), axis=-1)

    def transform(self, matrix):
        src_pos = vc.to_homogeneous_point(self._src_pos)
        src_pos = vc.to_vec(vc.matrix_transform(matrix, src_pos))

        det_vec = self._det_vec.transform(matrix)
        return ConeVectorGeometry(
            self.det_shape, src_pos, det_vec.det_pos, det_vec.det_v, det_vec.det_u
        )
