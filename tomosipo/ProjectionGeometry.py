import numpy as np
from .utils import up_tuple


def is_projection_geometry(g):
    return isinstance(g, ProjectionGeometry)


class ProjectionGeometry(object):
    """Documentation for ProjectionGeometry

    A general base class for projection geometries.
    """

    def __init__(self, shape=1):
        """Create a projection geometry

        :param shape:
        :returns:
        :rtype:

        """
        shape = up_tuple(shape, 2)
        # TODO: Handle case that shape is not an integer
        if not np.all(np.array(shape) > 0):
            raise ValueError("Shape must be strictly positive.")

        self.shape = shape

    def __repr__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def to_astra(self):
        raise NotImplementedError()

    def reshape(self, new_shape):
        """Reshape detector pixels


        :param new_shape: int or (int, int)
            The new shape of the detector in pixels in `v` (height)
            and `u` (width) direction.
        :returns: `self`
        :rtype: ProjectionGeometry

        """
        new_shape = up_tuple(new_shape, 2)
        self.shape = new_shape

        return self

    def to_vector(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        raise NotImplementedError()

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

        This function projects onto the detector even if the ray is
        moving away from the detector instead of towards it, or if the
        detector is between the source and point instead of behind the
        point.

        :param point: A three-dimensional vector (preferably np.array)
        :returns: np.array([[detector_intersection_y, detector_intersection_x],
                            .....])
        :rtype: np.array (num_angles * 2)
        """

        raise NotImplementedError()

    def is_cone(self):
        return self._is_cone

    def is_parallel(self):
        return self._is_parallel

    def is_vector(self):
        return self._is_vector

    def get_num_angles(self):
        """Return the number of angles in the projection geometry

        :returns:
            The number of angles in the projection geometry.
        :rtype: integer

        """
        raise NotImplementedError()

    @property
    def detector_sizes(self):
        """Returns a vector with the size of each detector

        :returns: np.array
            Array with shape (num_angles, 2) in v and u direction
            (height x width)
        :rtype: np.array

        """
        raise NotImplementedError()
