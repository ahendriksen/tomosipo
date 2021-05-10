import numpy as np
import tomosipo as ts
from tomosipo.types import ToShape2D, ToPos, ToSize2D, ToVec
import tomosipo.vector_calc as vc
from .base_projection import ProjectionGeometry
from . import det_vec as dv
from .transform import Transform


def cone_vec(
    *, shape: ToShape2D, src_pos: ToVec, det_pos: ToVec, det_v: ToVec, det_u: ToVec
):
    """Create an arbitrarily oriented cone-beam geometry

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

    Returns
    -------
    ConeVectorGeometry
        An arbitrarily oriented cone-beam geometry
    """
    return ConeVectorGeometry(
        shape=shape, src_pos=src_pos, det_pos=det_pos, det_v=det_v, det_u=det_u
    )


def random_cone_vec():
    """Generates a random cone vector geometry

    :returns: a random cone vector geometry
    :rtype: `ConeVectorGeometry`

    """
    T = ts.geometry.random_transform()
    return T * ts.geometry.random_cone().to_vec()


class ConeVectorGeometry(ProjectionGeometry):
    """Documentation for ConeVectorGeometry

    A class for representing cone vector geometries.
    """

    def __init__(self, *, shape, src_pos, det_pos, det_v, det_u):
        """Create an arbitrarily oriented cone-beam geometry

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
        """
        super(ConeVectorGeometry, self).__init__(shape=shape)

        src_pos = ts.types.to_vec(src_pos, "source position")
        det_pos = ts.types.to_vec(det_pos, "detector position")
        det_v = ts.types.to_vec(det_v, "v axis")
        det_u = ts.types.to_vec(det_u, "u axis")

        src_pos, det_pos, det_v, det_u = np.broadcast_arrays(
            src_pos, det_pos, det_v, det_u
        )

        shapes = [x.shape for x in [src_pos, det_pos, det_v, det_u]]

        if min(shapes) != max(shapes):
            raise ValueError(
                "Not all arguments src_pos, det_pos, det_v, det_u are the same shape. "
                f"Got: {shapes}"
            )

        self._src_pos = src_pos
        self._det_vec = dv.det_vec(shape, det_pos, det_v, det_u)

        self._is_cone = True
        self._is_parallel = False
        self._is_vector = True

    def __repr__(self):
        with ts.utils.print_options():
            return (
                f"ts.cone_vec(\n"
                f"    shape={self.det_shape},\n"
                f"    src_pos={repr(self._src_pos)},\n"
                f"    det_pos={repr(self._det_vec.det_pos)},\n"
                f"    det_v={repr(self._det_vec.det_v)},\n"
                f"    det_u={repr(self._det_vec.det_u)},\n"
                f")"
            )

    def __eq__(self, other):
        if not isinstance(other, ConeVectorGeometry):
            return False

        spos_diff = self._src_pos - other.src_pos

        return self._det_vec == other._det_vec and np.all(abs(spos_diff) < ts.epsilon)

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
        return ConeVectorGeometry(
            shape=det_vec.det_shape,
            src_pos=new_src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
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

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return ConeVectorGeometry(
            shape=shape,
            src_pos=src_pos[:, ::-1],
            det_pos=det_pos[:, ::-1],
            det_v=det_v[:, ::-1],
            det_u=det_u[:, ::-1],
        )

    def to_vec(self):
        return self

    def to_vol(self):
        """Returns a volume vector geometry representing the detector

        :returns: a volume vector geometry representing the detector
        :rtype:  `VolumeVectorGeometry`

        """
        return self._det_vec.to_vol()

    ###########################################################################
    #                                Properties                               #
    ###########################################################################

    @ProjectionGeometry.num_angles.getter
    def num_angles(self):
        return self._det_vec.num_angles

    @ProjectionGeometry.angles.getter
    def angles(self):
        raise NotImplementedError()

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

    @ProjectionGeometry.det_normal.getter
    def det_normal(self):
        return self._det_vec.det_normal

    @ProjectionGeometry.ray_dir.getter
    def ray_dir(self):
        raise NotImplementedError()

    @ProjectionGeometry.det_size.getter
    def det_size(self):
        return self._det_vec.det_size

    @ProjectionGeometry.det_sizes.getter
    def det_sizes(self):
        return self._det_vec.det_sizes

    @ProjectionGeometry.corners.getter
    def corners(self):
        return self._det_vec.corners

    @ProjectionGeometry.lower_left_corner.getter
    def lower_left_corner(self):
        return self._det_vec.lower_left_corner

    ###########################################################################
    #                                 Methods                                 #
    ###########################################################################

    def rescale_det(self, scale):
        det_vec = self._det_vec.rescale_det(scale)

        return ConeVectorGeometry(
            shape=det_vec.det_shape,
            src_pos=self.src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def reshape(self, new_shape):
        det_vec = self._det_vec.reshape(new_shape)
        return ConeVectorGeometry(
            shape=new_shape,
            src_pos=self.src_pos,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def project_point(self, point):
        if np.isscalar(point):
            point = ts.types.to_pos(point)

        v_origin = ts.types.to_vec(point)

        # TODO: Check v_origin shape (should be 3)

        det_pos = self._det_vec.det_pos
        det_v = self.det_v
        det_u = self.det_u
        det_normal = self.det_normal

        v_direction = v_origin - self._src_pos

        intersection = vc.intersect(v_origin, v_direction, det_pos, det_normal)

        det_i_u = vc.dot(intersection - det_pos, det_u) / vc.squared_norm(det_u)
        det_i_v = vc.dot(intersection - det_pos, det_v) / vc.squared_norm(det_v)

        return np.stack((det_i_v, det_i_u), axis=-1)

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
