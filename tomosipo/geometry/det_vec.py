"""Vector geometry containing just the detector

The class in this file can be encapsulated by ConeVectorGeometry and
ParallelVectorGeometry.

"""

import warnings
import numpy as np
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.utils import up_tuple, up_slice, slice_interval
from numbers import Integral
from .base_projection import ProjectionGeometry
from .transform import Transform


def det_vec(shape, det_pos, det_v, det_u):
    """Create a detector vector geometry

    :param shape: (`int`, `int`) or `int`
        The detector shape in pixels. If tuple, the order is
        (height, width). Otherwise, the pixel has the same number of
        pixels in the U and V direction.
    :param det_pos:
        A numpy array of dimension (num_positions, 3) with the
        detector center positions in (Z, Y, X) order.
    :param det_v:
        A numpy array of dimension (num_positions, 3) with the
        vector pointing from the detector (0, 0) to (1, 0) pixel
        (up).
    :param det_u:
        A numpy array of dimension (num_positions, 3) with the
        vector pointing from the detector (0, 0) to (0, 1) pixel
        (sideways).
    :returns:
    :rtype:

    """
    return DetectorVectorGeometry(shape, det_pos, det_v, det_u)


def random_det_vec():
    """Generates a random cone vector geometry

    :returns: a random cone vector geometry
    :rtype: `ConeVectorGeometry`

    """
    shape = np.random.uniform(10, 20, size=2).astype(np.int)
    num_angles = int(np.random.uniform(1, 100))
    det_pos = np.random.normal(size=(num_angles, 3))
    det_v = np.random.normal(size=(num_angles, 3))
    det_u = np.random.normal(size=(num_angles, 3))

    return det_vec(shape, det_pos, det_v, det_u)


class DetectorVectorGeometry(ProjectionGeometry):
    """A class for representing detector vector geometries.

    This class is a helper for the ConeVectorGeometry and
    ParallelVectorGeometry. It represents the detector part of these
    geometries. It does not have any references to a source or ray
    directions.

    """

    def __init__(self, shape, det_pos, det_v, det_u):
        """Create a detector vector geometry

        :param shape: (`int`, `int`) or `int`
            The detector shape in pixels. If tuple, the order is
            (height, width). Else the pixel has the same number of
            pixels in the U and V direction.
        :param det_pos:
            A numpy array of dimension (num_positions, 3) with the
            detector center positions in (Z, Y, X) order.
        :param det_v:
            A numpy array of dimension (num_positions, 3) with the
            vector pointing from the detector (0, 0) to (1, 0) pixel
            (up).
        :param det_u:
            A numpy array of dimension (num_positions, 3) with the
            vector pointing from the detector (0, 0) to (0, 1) pixel
            (sideways).
        :returns:
        :rtype:

        """
        super(DetectorVectorGeometry, self).__init__(shape=shape)

        det_pos, det_v, det_u = (vc.to_vec(x) for x in (det_pos, det_v, det_u))
        det_pos, det_v, det_u = np.broadcast_arrays(det_pos, det_v, det_u)

        vc.check_same_shapes(det_pos, det_v, det_u)

        self._det_pos = det_pos
        self._det_v = det_v
        self._det_u = det_u

        self._is_cone = False
        self._is_parallel = False
        self._is_vector = True

    def __repr__(self):
        return (
            f"DetectorVectorGeometry(\n"
            f"    shape={self.det_shape},\n"
            f"    det_pos={self._det_pos},\n"
            f"    det_u={self._det_v},\n"
            f"    det_v={self._det_u}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, DetectorVectorGeometry):
            return False

        dpos_diff = self._det_pos - other._det_pos
        us_diff = self._det_u - other._det_u
        vs_diff = self._det_v - other._det_v

        return (
            self.det_shape == other.det_shape
            and np.all(abs(dpos_diff) < ts.epsilon)
            and np.all(abs(us_diff) < ts.epsilon)
            and np.all(abs(vs_diff) < ts.epsilon)
        )

    def __getitem__(self, key):
        """Slice the geometry to create a sub-geometry

        This geometry can be sliced in the number of projections:

        >>> ts.geometry.random_det_vec()[0::2]

        Or in the detector plane:

        >>> ts.geometry.random_det_vec()[:, ::2, ::2]

        :param key:
        :returns:
        :rtype:

        """
        full_slice = slice(None, None, None)

        if isinstance(key, Integral) or isinstance(key, slice):
            key = (key, full_slice, full_slice)
        while isinstance(key, tuple) and len(key) < 3:
            key = (*key, full_slice)

        if isinstance(key, tuple) and len(key) == 3:
            v0, v1, lenV, stepV = slice_interval(
                0, self.det_shape[0], self.det_shape[0], key[1]
            )
            u0, u1, lenU, stepU = slice_interval(
                0, self.det_shape[1], self.det_shape[1], key[2]
            )
            # Calculate new lower-left corner, top-right corner, and center.
            new_llc = self.lower_left_corner + v0 * self.det_v + u0 * self.det_u
            new_trc = self.lower_left_corner + v1 * self.det_v + u1 * self.det_u
            new_center = (new_llc + new_trc) / 2

            new_shape = (lenV, lenU)
            new_det_pos = new_center[up_slice(key[0])]
            new_det_vs = self._det_v[up_slice(key[0])] * stepV
            new_det_us = self._det_u[up_slice(key[0])] * stepU

            return det_vec(new_shape, new_det_pos, new_det_vs, new_det_us)

    def to_astra(self):
        row_count, col_count = self.det_shape
        # We do not have ray_dir or src_pos, so we just set the first
        # three columns to zero.
        vectors = np.concatenate(
            [
                self._det_pos[:, ::-1] * 0,
                self._det_pos[:, ::-1],
                self._det_u[:, ::-1],
                self._det_v[:, ::-1],
            ],
            axis=1,
        )

        return {
            "type": "det_vec",  # Astra does not support this type
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "Vectors": vectors,
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "det_vec":
            raise ValueError(
                "DetectorVectorGeometry.from_astra only supports 'det_vec' type astra geometries."
            )

        vecs = astra_pg["Vectors"]
        # detector pos:
        det_pos = vecs[:, 3:6]
        # Detector u and v direction
        det_u = vecs[:, 6:9]
        det_v = vecs[:, 9:12]

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return det_vec(shape, det_pos[:, ::-1], det_v[:, ::-1], det_u[:, ::-1])

    def to_vec(self):
        return self

    def to_box(self):
        """Returns a box representating the detector

        :returns: detector_box
        :rtype:  `OrientedBox`

        """
        det_pos = self._det_pos
        w = self._det_v  # v points up, w points up
        u = self._det_u  # detector_u and u point in the same direction

        # TODO: Fix vc.norm so we do not need [:, None]
        # We do not want to introduce scaling, so we normalize w and u.
        w = w / vc.norm(w)[:, None]
        u = u / vc.norm(u)[:, None]
        # This is the detector normal and has norm 1. In right-handed
        # coordinates, it would point towards the source usually. Now
        # it points "into" the detector.
        v = self.det_normal
        v = v / vc.norm(v)[:, None]

        # TODO: Warn when detector size changes during rotation.
        det_height, det_width = self.det_sizes[0]

        if np.any(abs(np.ptp(self.det_sizes, axis=0)) > ts.epsilon):
            warnings.warn(
                "The detector size is not uniform. "
                "Using first detector size for the box. "
                "To inhibit this warning, please make sure that the absolute size of the detector does not change. "
            )

        det_box = ts.box((det_height, 0, det_width), det_pos, w, v, u)

        return det_box

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @ProjectionGeometry.num_angles.getter
    def num_angles(self):
        return len(self._det_pos)

    @ProjectionGeometry.angles.getter
    def angles(self):
        raise NotImplementedError()

    @ProjectionGeometry.src_pos.getter
    def src_pos(self):
        raise NotImplementedError()

    @ProjectionGeometry.det_pos.getter
    def det_pos(self):
        return np.copy(self._det_pos)

    @ProjectionGeometry.det_v.getter
    def det_v(self):
        return np.copy(self._det_v)

    @ProjectionGeometry.det_u.getter
    def det_u(self):
        return np.copy(self._det_u)

    @ProjectionGeometry.det_normal.getter
    def det_normal(self):
        return vc.cross_product(self._det_u, self.det_v)

    @ProjectionGeometry.ray_dir.getter
    def ray_dir(self):
        raise NotImplementedError()

    @ProjectionGeometry.det_size.getter
    def det_size(self):
        height = vc.norm(self._det_v * self.det_shape[0])
        width = vc.norm(self._det_u * self.det_shape[1])

        height_constant = abs(min(height) - max(height)) < ts.epsilon
        width_constant = abs(min(width) - max(width)) < ts.epsilon

        if height_constant and width_constant:
            return (height[0], width[0])
        else:
            raise ValueError("The size of the detector is not constant. ")

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        height = vc.norm(self._det_v * self.det_shape[0])
        width = vc.norm(self._det_u * self.det_shape[1])
        return np.stack([height, width], axis=1)

    @ProjectionGeometry.corners.getter
    def corners(self):
        ds = self._det_pos
        u_offset = self._det_u * self.det_shape[1] / 2
        v_offset = self._det_v * self.det_shape[0] / 2

        return np.array(
            [
                ds - u_offset - v_offset,
                ds - u_offset + v_offset,
                ds + u_offset - v_offset,
                ds + u_offset + v_offset,
            ]
        ).transpose([1, 0, 2])

    @ProjectionGeometry.lower_left_corner.getter
    def lower_left_corner(self):
        return (
            self._det_pos
            - (self._det_v * self.det_shape[0]) / 2
            - (self._det_u * self.det_shape[1]) / 2
        )

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
        scaleV, scaleU = up_tuple(scale, 2)
        scaleV, scaleU = int(scaleV), int(scaleU)

        shape = (self.det_shape[0] // scaleV, self.det_shape[1] // scaleU)
        det_v = self._det_v * scaleV
        det_u = self._det_u * scaleU

        return det_vec(shape, self._det_pos, det_v, det_u)

    def reshape(self, new_shape):
        """Reshape detector pixels without changing detector size


        :param new_shape: int or (int, int)
            The new shape of the detector in pixels in `v` (height)
            and `u` (width) direction.
        :returns: `self`
        :rtype: ProjectionGeometry

        """
        new_shape = up_tuple(new_shape, 2)
        det_v = self.det_v / new_shape[0] * self.det_shape[0]
        det_u = self.det_u / new_shape[1] * self.det_shape[1]

        return det_vec(new_shape, self.det_pos, det_v, det_u)

    def project_point(self, point):
        raise NotImplementedError()

    def __rmul__(self, other):
        if isinstance(other, Transform):
            matrix = other.matrix
            det_pos = vc.to_homogeneous_point(self._det_pos)
            det_v = vc.to_homogeneous_vec(self._det_v)
            det_u = vc.to_homogeneous_vec(self._det_u)

            det_pos = vc.to_vec(vc.matrix_transform(matrix, det_pos))
            det_v = vc.to_vec(vc.matrix_transform(matrix, det_v))
            det_u = vc.to_vec(vc.matrix_transform(matrix, det_u))

            return det_vec(self.det_shape, det_pos, det_v, det_u)
