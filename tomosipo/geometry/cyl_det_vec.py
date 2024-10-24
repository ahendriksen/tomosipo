"""Vector geometry containing just the cylindrical detector

The class in this file can be encapsulated by CylConeVectorGeometry and
CylParallelVectorGeometry.

"""

import numpy as np
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.utils import slice_interval
from numbers import Integral
from .base_projection import ProjectionGeometry
from .transform import Transform
from .det_vec import DetectorVectorGeometry


def cyl_det_vec(shape, det_pos, det_v, det_u, curvature):
    """Create a cylindrical detector vector geometry

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
    :param curvature:
        A float scalar describing detector curvature.
    :returns:
    :rtype:

    """
    return CylDetectorVectorGeometry(shape, det_pos, det_v, det_u, curvature)


def random_cyl_det_vec():
    raise NotImplementedError


class CylDetectorVectorGeometry(DetectorVectorGeometry):
    """A class for representing cylindrical detector vector geometries.

    This class is a helper for the CylConeVectorGeometry and
    CylParallelVectorGeometry. It represents the detector part of these
    geometries. It does not have any references to a source or ray
    directions.

    """

    def __init__(self, shape, det_pos, det_v, det_u, curvature=0.0):
        """Create a cylindrical detector vector geometry.

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
        :param curvature:
            A float scalar describing detector curvature. Default is 0.
        :returns:
        :rtype:

        """
        super(CylDetectorVectorGeometry, self).__init__(shape, det_pos, det_v, det_u)
        self._curvature = curvature


    def __repr__(self):
        return (
            f"DetectorVectorGeometry(\n"
            f"    shape={self.det_shape},\n"
            f"    det_pos={self._det_pos},\n"
            f"    det_u={self._det_v},\n"
            f"    det_v={self._det_u}"
            f"    curvature={self._curvature}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, CylDetectorVectorGeometry):
            return False

        return (
            super().__eq__(self, other)
            and abs(self._curvature - other._curvature) < ts.epsilon
        )

    def __getitem__(self, key):
        raise NotImplementedError

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
                np.repeat(self.curvature, self.num_angles)
            ],
            axis=1,
        )

        return {
            "type": "cyl_det_vec",  # Astra does not support this type
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "Vectors": vectors,
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "cyl_det_vec":
            raise ValueError(
                "DetectorVectorGeometry.from_astra only supports 'det_vec' type astra geometries."
            )

        vecs = astra_pg["Vectors"]
        # detector pos:
        det_pos = vecs[:, 3:6]
        # Detector u and v direction
        det_u = vecs[:, 6:9]
        det_v = vecs[:, 9:12]
        curvature = vecs[0, 12]
        assert len(np.unique(vecs[:, 12])) == 1, "Non-unique curvature values per geometry are not supported"

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return cyl_det_vec(shape, det_pos[:, ::-1], det_v[:, ::-1], det_u[:, ::-1], curvature)

    def to_vec(self):
        return self

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @property
    def curvature(self):
        return self._curvature

    @ProjectionGeometry.det_size.getter
    def det_size(self):
        raise NotImplementedError

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        raise NotImplementedError

    @ProjectionGeometry.corners.getter
    def corners(self):
        raise NotImplementedError

    @ProjectionGeometry.lower_left_corner.getter
    def lower_left_corner(self):
        raise NotImplementedError

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################

    def rescale_det(self, scale):
        raise NotImplementedError

    def reshape(self, new_shape):
        raise NotImplementedError

    def project_point(self, point):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError
