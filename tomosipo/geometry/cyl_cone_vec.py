import numpy as np
import tomosipo as ts
from tomosipo.types import ToShape2D, ToVec, ToScalars
import tomosipo.vector_calc as vc
from .base_projection import ProjectionGeometry
from . import cyl_det_vec as cdv
from .transform import Transform
from .cone_vec import ConeVectorGeometry


def cyl_cone_vec(
    *, shape: ToShape2D, src_pos: ToVec, det_pos: ToVec,
    det_v: ToVec, det_u: ToVec, curvature: ToScalars
):
    """Create an arbitrarily oriented cone-beam geometry with a cylindrical detector.

    Parameters
    ----------

    shape:
        The detector shape in pixels. If tuple, the order is
        `(height, width)`. Else the pixel has the same number of
        pixels in the `u` and `v` direction.

    src_pos:
        A numpy array of dimension `(num_positions, 3)` with the
        source positions in `(z, y, x)` order.

    det_pos:
        A numpy array of dimension `(num_positions, 3)` with the
        detector center positions in `(z, y, x)` order.

    det_v:
        A numpy array of dimension `(num_positions, 3)` with the vector pointing
        from the detector `(0, 0)` to `(1, 0)` pixel (up).

    det_u:
        A numpy array of dimension `(num_positions, 3)` with the
        vector pointing from the detector `(0, 0)` to `(0, 1)` pixel
        (sideways).

    curvature:
        A float scalar specifying detector curvature.

    Returns
    -------
    CylConeVectorGeometry
        An arbitrarily oriented cone-beam geometry with a cylindrical detector.
    """
    return CylConeVectorGeometry(
        shape=shape, src_pos=src_pos, det_pos=det_pos, det_v=det_v, det_u=det_u,
        curvature=curvature
    )


class CylConeVectorGeometry(ConeVectorGeometry):
    """Documentation for CylConeVectorGeometry

    A class for representing cone vector geometries with cylindrical detectors.
    """

    def __init__(self, *, shape, src_pos, det_pos, det_v, det_u, curvature):
        """Create an arbitrarily oriented cone-beam geometry with cylindrical detector.

        Parameters
        ----------

        shape:
            The detector shape in pixels. If tuple, the order is
            `(height, width)`. Else the pixel has the same number of
            pixels in the `u` and `v` direction.

        src_pos:
            A numpy array of dimension `(num_positions, 3)` with the
            source positions in `(z, y, x)` order.

        det_pos:
            A numpy array of dimension `(num_positions, 3)` with the
            detector center positions in `(z, y, x)` order.

        det_v:
            A numpy array of dimension `(num_positions, 3)` with the vector pointing
            from the detector `(0, 0)` to `(1, 0)` pixel (up).

        det_u:
            A numpy array of dimension `(num_positions, 3)` with the
            vector pointing from the detector `(0, 0)` to `(0, 1)` pixel
            (sideways).

        curvature:
            A float scalar specifying detector curvature.
        """
        super().__init__(shape=shape, src_pos=src_pos, det_pos=det_pos, det_v=det_v, det_u=det_u)
        self._det_vec = cdv.cyl_det_vec(shape, det_pos, det_v, det_u, curvature)

    def __repr__(self):
        with ts.utils.print_options():
            return (
                f"ts.cy_cone_vec(\n"
                f"    shape={self.det_shape},\n"
                f"    src_pos={repr(self._src_pos)},\n"
                f"    det_pos={repr(self._det_vec.det_pos)},\n"
                f"    det_v={repr(self._det_vec.det_v)},\n"
                f"    det_u={repr(self._det_vec.det_u)},\n"
                f"    curvature={self._det_vec.curvature}\n"
                f")"
            )

    def __eq__(self, other):
        if not isinstance(other, CylConeVectorGeometry):
            return False
        return super().__eq__(self, other)

    def __getitem__(self, key):
        """Slice the geometry to create a sub-geometry

        This geometry can be sliced by angle. The following obtains a
        sub-geometry containing every second projection:

        >>> ts.cone(angles=10, cone_angle=1/2).to_vec()[0::2].num_angles
        5

        This geometry can also be sliced in the detector plane:

        >>> ts.cone(shape=10, cone_angle=1/2).to_vec()[:, ::2, ::2].det_shape
        (5, 5)

        :param key:
        :returns:
        :rtype:

        """

        det_vec = self._det_vec[key]
        if isinstance(key, tuple):
            key, *_ = key

        new_src_pos = self._src_pos[key]
        return CylConeVectorGeometry(
            shape=det_vec.det_shape,
            src_pos=new_src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
            curvature=self.curvature
        )

    def to_astra(self):
        row_count, col_count = self.det_shape
        vectors = np.concatenate(
            [
                self._src_pos[:, ::-1],
                self._det_vec.det_pos[:, ::-1],
                self._det_vec.det_u[:, ::-1],
                self._det_vec.det_v[:, ::-1],
                self._det_vec.curvature * np.ones([self.num_angles, 1])
            ],
            axis=1,
        )

        return {
            "type": "cyl_cone_vec",
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "Vectors": vectors,
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "cyl_cone_vec":
            raise ValueError(
                "ConeVectorGeometry.from_astra only supports 'cone_vec' type astra geometries."
            )

        vecs = astra_pg["Vectors"]
        # ray direction (parallel) / source_position (cone)
        src_pos = vecs[:, :3]
        # detector pos:
        det_pos = vecs[:, 3:6]
        # Detector u and v direction
        det_u = vecs[:, 6:9]
        det_v = vecs[:, 9:12]
        curvature = vecs[0, 12]
        assert len(np.unique(vecs[:, 12])) == 1, "Non-unique curvature values per geometry are not supported"

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return CylConeVectorGeometry(
            shape=shape,
            src_pos=src_pos[:, ::-1],
            det_pos=det_pos[:, ::-1],
            det_v=det_v[:, ::-1],
            det_u=det_u[:, ::-1],
            curvature=curvature
        )

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @property
    def curvature(self):
        return self._det_vec.curvature

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################

    def rescale_det(self, scale):
        det_vec = self._det_vec.rescale_det(scale)
        return CylConeVectorGeometry(
            shape=det_vec.det_shape,
            src_pos=self.src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def reshape(self, new_shape):
        det_vec = self._det_vec.reshape(new_shape)
        return CylConeVectorGeometry(
            shape=new_shape,
            src_pos=self.src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def __rmul__(self, other):
        if isinstance(other, Transform):
            src_pos = other.transform_point(self._src_pos)

            det_vec = other * self._det_vec
            return ConeVectorGeometry(
                shape=self.det_shape,
                src_pos=src_pos,
                det_pos=det_vec.det_pos,
                det_v=det_vec.det_v,
                det_u=det_vec.det_u,
            )
