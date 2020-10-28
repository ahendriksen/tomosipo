"""This module implements a parallel-beam vector projection geometry

"""

import numpy as np
import tomosipo as ts
import tomosipo.vector_calc as vc
from tomosipo.utils import up_tuple, up_slice
from .base_projection import ProjectionGeometry
from . import det_vec as dv
from .transform import Transform


def parallel_vec(*, shape, ray_dir, det_pos, det_v, det_u):
    """Create a parallel-beam vector geometry

    :param shape: (`int`, `int`) or `int`
        The detector shape in pixels. If tuple, the order is
        (height, width). Else the pixel has the same number of
        pixels in the U and V direction.
    :param ray_dir: np.array
        A numpy array of dimension (num_positions, 3) with the
        ray direction in (Z, Y, X) order.
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

    :returns: An arbitrarily oriented parallel-beam geometry
    :rtype: ParallelVectorGeometry
    """
    return ParallelVectorGeometry(shape, ray_dir, det_pos, det_v, det_u)


def random_parallel_vec():
    num_angles = int(np.random.uniform(1, 20))
    return parallel_vec(
        shape=np.random.uniform(10, 20, size=2).astype(np.int),
        ray_dir=np.random.normal(size=(num_angles, 3)),
        det_pos=np.random.normal(size=(num_angles, 3)),
        det_v=np.random.normal(size=(num_angles, 3)),
        det_u=np.random.normal(size=(num_angles, 3)),
    )


class ParallelVectorGeometry(ProjectionGeometry):
    """Documentation for ParallelVectorGeometry

    """

    def __init__(self, shape, ray_dir, det_pos, det_v, det_u):
        """Create a parallel-beam vector geometry

        :param shape: (`int`, `int`) or `int`
            The detector shape in pixels. If tuple, the order is
            (height, width). Else the pixel has the same number of
            pixels in the U and V direction.
        :param ray_dir: np.array
            A numpy array of dimension (num_positions, 3) with the
            ray direction in (Z, Y, X) order.
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
        :returns: An arbitrarily oriented parallel-beam geometry
        :rtype: ParallelVectorGeometry
        """

        super(ParallelVectorGeometry, self).__init__(shape=shape)

        ray_dir, det_pos, det_v, det_u = (
            vc.to_vec(x) for x in (ray_dir, det_pos, det_v, det_u)
        )
        ray_dir, det_pos, det_v, det_u = np.broadcast_arrays(
            ray_dir, det_pos, det_v, det_u
        )

        vc.check_same_shapes(ray_dir, det_pos, det_v, det_u)

        self._ray_dir = ray_dir
        self._det_vec = dv.det_vec(shape, det_pos, det_v, det_u)

        self._is_cone = False
        self._is_parallel = True
        self._is_vector = True

    def __repr__(self):
        with ts.utils.print_options():
            return (
                f"ts.parallel_vec(\n"
                f"    shape={self.det_shape},\n"
                f"    ray_dir={repr(self._ray_dir)},\n"
                f"    det_pos={repr(self._det_vec.det_pos)},\n"
                f"    det_v={repr(self._det_vec.det_v)},\n"
                f"    det_u={repr(self._det_vec.det_u)},\n"
                f")"
            )

    def __eq__(self, other):
        if not isinstance(other, ParallelVectorGeometry):
            return False

        dray_dir = self._ray_dir - other._ray_dir

        return self._det_vec == other._det_vec and np.all(abs(dray_dir) < ts.epsilon)

    def __getitem__(self, key):
        """Slice the geometry to create a sub-geometry

        This geometry can be sliced by angle. The following obtains a
        sub-geometry containing every second projection:

        >>> ts.parallel(angles=10).to_vec()[0::2]

        This geometry can also be sliced in the detector plane:

        >>> ts.parallel(shape=10).to_vec()[:, ::2, ::2]

        :param key:
        :returns:
        :rtype:

        """
        det_vec = self._det_vec[key]
        if isinstance(key, tuple):
            key, *_ = key

        new_ray_dir = self._ray_dir[key]
        return parallel_vec(
            shape=det_vec.det_shape,
            ray_dir=new_ray_dir,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def to_astra(self):
        row_count, col_count = self.det_shape
        vectors = np.concatenate(
            [
                self._ray_dir[:, ::-1],
                self._det_vec.det_pos[:, ::-1],
                self._det_vec.det_u[:, ::-1],
                self._det_vec.det_v[:, ::-1],
            ],
            axis=1,
        )

        return {
            "type": "parallel3d_vec",
            "DetectorRowCount": row_count,
            "DetectorColCount": col_count,
            "Vectors": vectors,
        }

    def from_astra(astra_pg):
        if astra_pg["type"] != "parallel3d_vec":
            raise ValueError(
                "ParallelVectorGeometry.from_astra only supports 'parallel3d_vec' type astra geometries."
            )

        vecs = astra_pg["Vectors"]
        # ray direction (parallel) / source_position (cone)
        ray_dir = vecs[:, :3]
        # detector pos:
        det_pos = vecs[:, 3:6]
        # Detector u and v direction
        det_u = vecs[:, 6:9]
        det_v = vecs[:, 9:12]

        shape = (astra_pg["DetectorRowCount"], astra_pg["DetectorColCount"])
        return parallel_vec(
            shape=shape,
            ray_dir=ray_dir[:, ::-1],
            det_pos=det_pos[:, ::-1],
            det_v=det_v[:, ::-1],
            det_u=det_u[:, ::-1],
        )

    def to_vec(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        return self

    def to_box(self):
        """Returns an oriented box representating the detector

        :returns: an oriented box representating the detector
        :rtype:  `VolumeVectorGeometry`

        """
        return self._det_vec.to_box()

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
        raise NotImplementedError()

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
        return np.copy(self._ray_dir)

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

        return parallel_vec(
            shape=det_vec.det_shape,
            ray_dir=self.ray_dir,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def reshape(self, new_shape):
        det_vec = self._det_vec.reshape(new_shape)
        return parallel_vec(
            shape=new_shape,
            ray_dir=self.ray_dir,
            det_pos=det_vec.det_pos,
            det_v=det_vec.det_v,
            det_u=det_vec.det_u,
        )

    def project_point(self, point):
        if np.isscalar(point):
            point = up_tuple(point, 3)

        v_origin = vc.to_vec(point)

        det_pos = self._det_vec.det_pos
        det_v = self.det_v
        det_u = self.det_u
        det_normal = self.det_normal
        v_direction = self._ray_dir

        intersection = vc.intersect(v_origin, v_direction, det_pos, det_normal)

        det_i_u = vc.dot(intersection - det_pos, det_u) / vc.squared_norm(det_u)
        det_i_v = vc.dot(intersection - det_pos, det_v) / vc.squared_norm(det_v)

        return np.stack((det_i_v, det_i_u), axis=-1)

    def __rmul__(self, other):
        if isinstance(other, Transform):
            ray_dir = other.transform_vec(self._ray_dir)

            det_vec = other * self._det_vec
            return parallel_vec(
                shape=self.det_shape,
                ray_dir=ray_dir,
                det_pos=det_vec.det_pos,
                det_v=det_vec.det_v,
                det_u=det_vec.det_u,
            )
