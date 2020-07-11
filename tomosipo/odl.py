import warnings
from packaging import version
import astra
try:
    import odl
except ModuleNotFoundError:
    warnings.warn(
        "\n------------------------------------------------------------\n\n"
        "Cannot import ODL. \n"
        "Please make sure to install ODL. \n"
        "You can install ODL using: \n\n"
        " > conda install -c odlgroup odl \n"
        "Or: \n"
        " > pip install git+https://github.com/odlgroup/odl \n"
        "\n------------------------------------------------------------\n\n"
    )
    raise

import tomosipo as ts
from odl.tomo.backends.astra_setup import (
    astra_projection_geometry,
    astra_volume_geometry
)


def from_odl(geom_or_op):
    is_geometry = isinstance(geom_or_op, odl.tomo.geometry.Geometry)
    is_volume = isinstance(geom_or_op, odl.discr.DiscretizedSpace)

    if is_geometry:
        # convert fan beam and 2D parallel beam geometries to 3D geometries
        fan_geometry_cls = (
            odl.tomo.FanBeamGeometry
            if version.parse(odl.__version__) > version.parse('0.7.0') else
            odl.tomo.FanFlatGeometry)
        if isinstance(geom_or_op, fan_geometry_cls):
            geom_or_op = fan_to_cone_beam_geometry(geom_or_op)
        if isinstance(geom_or_op, odl.tomo.Parallel2dGeometry):
            geom_or_op = parallel_2d_to_3d_geometry(geom_or_op)
        return ts.from_astra(astra_projection_geometry(geom_or_op))

    if is_volume:
        if geom_or_op.ndim == 2:
            geom_or_op = discretized_space_2d_to_3d(geom_or_op)
        return ts.from_astra(astra_volume_geometry(geom_or_op))

    raise TypeError(
        f"The object of type {type(geom_or_op)} cannot be imported from ODL. "
    )


# TODO: implement converson of tomosipo geometries to ODL geometries.
def to_odl(geom_or_op):
    pass


def fan_to_cone_beam_geometry(fb_geometry, det_z_span=1.):
    """
    Convert 2D `odl.tomo.FanBeamGeometry` to 3D `odl.tomo.ConeBeamGeometry`,
    by adding a trivial third axis.
    
    For odl<='0.7.0' the classes are `odl.tomo.FanFlatGeometry` and
    `odl.tomo.ConeFlatGeometry`.

    Parameters
    ----------
    fb_geometry : `odl.tomo.FanBeamGeometry`
        The fan beam geometry to be converted.
    det_z_span : float, optional
        Size of the detector in the trivial third dimension.
        The default is ``1.``.

    Returns
    -------
    cb_geometry : `odl.tomo.ConeBeamGeometry`
        The cone beam geometry that coincides with the given fan beam geometry
        at the central slice (third coordinate equals zero).
    """
    if version.parse(odl.__version__) <= version.parse('0.7.0'):
        if not isinstance(fb_geometry, odl.tomo.FanFlatGeometry):
            raise TypeError('expected a `FanFlatGeometry`')
    else:
        if not isinstance(fb_geometry, odl.tomo.FanBeamGeometry):
            raise TypeError('expected a `FanBeamGeometry`')
    apart = fb_geometry.motion_partition
    det_min_pt = [fb_geometry.det_partition.min_pt[0], -det_z_span/2.]
    det_max_pt = [fb_geometry.det_partition.max_pt[0], +det_z_span/2.]
    dpart = odl.discr.RectPartition(
        odl.set.IntervalProd(det_min_pt, det_max_pt),
        odl.discr.RectGrid(fb_geometry.det_partition.coord_vectors[0], [0.]))
    src_radius = fb_geometry.src_radius
    det_radius = fb_geometry.det_radius
    if version.parse(odl.__version__) > version.parse('0.7.0'):
        det_curvature_radius = (
            None if fb_geometry.det_curvature_radius is None
            else (fb_geometry.det_curvature_radius, None))
    src_to_det_init = [fb_geometry.src_to_det_init[0],
                       fb_geometry.src_to_det_init[1],
                       0.]
    det_axes_init = ([fb_geometry.det_axis_init[0],
                      fb_geometry.det_axis_init[1],
                      0.],
                     [0., 0., 1.])
    translation = [fb_geometry.translation[0],
                   fb_geometry.translation[1],
                   0.]
    if version.parse(odl.__version__) <= version.parse('0.7.0'):
        cb_geometry = odl.tomo.ConeFlatGeometry(
            apart, dpart, src_radius, det_radius,
            pitch=0.,
            axis=[0., 0., 1.],
            src_to_det_init=src_to_det_init,
            det_axes_init=det_axes_init,
            translation=translation,
            check_bounds=fb_geometry.check_bounds
            )
        return cb_geometry
    cb_geometry = odl.tomo.ConeBeamGeometry(
        apart, dpart, src_radius, det_radius,
        det_curvature_radius=det_curvature_radius,
        pitch=0.,
        axis=[0., 0., 1.],
        src_to_det_init=src_to_det_init,
        det_axes_init=det_axes_init,
        translation=translation,
        check_bounds=fb_geometry.check_bounds
        )
    return cb_geometry


def parallel_2d_to_3d_geometry(pb2d_geometry, det_z_shape=1, det_z_span=None):
    """
    Convert `odl.tomo.Parallel2dGeometry` to `odl.tomo.Parallel3dAxisGeometry`,
    by adding a trivial third axis.

    Parameters
    ----------
    pb2d_geometry : `odl.tomo.Parallel2dGeometry`
        The 2d parallel beam geometry to be converted.
    det_z_shape : int, optional
        Number of detector pixels in the third dimension.
        Can be used to batch multiple 2d projections into one 3d projection.
    det_z_span : float or 2-tuple of float, optional
        Minimum and maximum point of the third axis (like passed to
        `uniform_partition`).
        If a single float is passed, ``(-det_z_span/2., det_z_span/2.)`` is
        used.
        If `None` (the default), ``(-det_z_shape/2., det_z_shape/2.)`` is used,
        resulting in a partition with cell side ``1.``.

    Returns
    -------
    pb3d_geometry : `odl.tomo.Parallel3dAxisGeometry`
        A 3d parallel beam geometry that coincides with the given 2d geometry
        in every slice (third coordinate fixed).
    """
    if not isinstance(pb2d_geometry, odl.tomo.Parallel2dGeometry):
        raise TypeError('expected a `Parallel2dGeometry`')
    if (version.parse(odl.__version__) <= version.parse('0.7.0') and
            version.parse(astra.__version__) > version.parse('1.9.0.dev')):
        warnings.warn(
            'versions of odl and astra are incompatible, the '
            'RayTransform for the 3d geometry will scale correctly in '
            'contrast to the 2d geometry, which is multiplied by the factor '
            '``1. / pb2d_geometry.det_partition.cell_volume``.')
    apart = pb2d_geometry.motion_partition
    if det_z_span is None:
        det_z_span = (-det_z_shape/2., det_z_shape/2.)
    elif isinstance(det_z_span, float):
        det_z_span = (-det_z_span/2., det_z_span/2.)
    det_z_partition = odl.uniform_partition(
        min_pt=det_z_span[0], max_pt=det_z_span[1], shape=det_z_shape)
    det_min_pt = [pb2d_geometry.det_partition.min_pt[0],
                  det_z_partition.min_pt[0]]
    det_max_pt = [pb2d_geometry.det_partition.max_pt[0],
                  det_z_partition.max_pt[0]]
    dpart = odl.discr.RectPartition(
        odl.set.IntervalProd(det_min_pt, det_max_pt),
        odl.discr.RectGrid(pb2d_geometry.det_partition.coord_vectors[0],
                           det_z_partition.coord_vectors[0]))
    # The RayTransform in 2d seems to act agnostic of the parameters
    # `det_pos_init`, `det_axes_init` and `translation`.
    # Seems to me like a bug, could be related to
    # https://github.com/odlgroup/odl/issues/359),
    # The 3d transform would handle them, so the results will be inconsistent.
    # Therefore i also did not test the following definitions.
    det_pos_init = [pb2d_geometry.det_pos_init[0],
                    pb2d_geometry.det_pos_init[1],
                    0.]
    det_axes_init = ([pb2d_geometry.det_axis_init[0],
                      pb2d_geometry.det_axis_init[1],
                      0.],
                      [0., 0., 1.])
    translation = [pb2d_geometry.translation[0],
                   pb2d_geometry.translation[1],
                   0.]
    pb3d_geometry = odl.tomo.Parallel3dAxisGeometry(
        apart, dpart,
        axis=[0., 0., 1.],
        det_pos_init=det_pos_init,
        det_axes_init=det_axes_init,
        translation=translation,
        check_bounds=pb2d_geometry.check_bounds
        )
    return pb3d_geometry


def discretized_space_2d_to_3d(space, z_shape=1, z_span=None):
    """
    Convert 2D `odl.discr.DiscretizedSpace` to 3D by adding a third axis.

    Parameters
    ----------
    space : `odl.discr.DiscretizedSpace`
        2D space to be converted.
    z_shape : int, optional
        Number of points in the third axis.
    z_span : float, optional
        Minimum and maximum point of the third axis (like passed to
        `uniform_partition`).
        If a single float is passed, ``(-z_span/2., z_span/2.)`` is
        used.
        If `None` (the default), ``(-z_shape/2., z_shape/2.)`` is used,
        resulting in a partition with cell side ``1.``.

    Returns
    -------
    space3d : `odl.discr.DiscretizedSpace`
        The 3D space that contains the given 2D space as a single
        third-axis-slice.
    """
    if z_span is None:
        z_span = (-z_shape / 2., z_shape / 2.)
    elif isinstance(z_span, float):
        z_span = (-z_span / 2., z_span / 2.)
    z_partition = odl.uniform_partition(
        min_pt=z_span[0], max_pt=z_span[1], shape=z_shape)
    rect_partition = odl.discr.RectPartition(
            odl.set.IntervalProd(
                [space.min_pt[0],
                 space.min_pt[1],
                 z_partition.min_pt[0]],
                [space.max_pt[0],
                 space.max_pt[1],
                 z_partition.max_pt[0]]),
            odl.discr.RectGrid(space.partition.coord_vectors[0],
                               space.partition.coord_vectors[1],
                               z_partition.coord_vectors[0]))
    tensor_space = odl.space.space_utils.tensor_space(
            (space.tspace.shape[0], space.tspace.shape[1], z_shape),
            dtype=space.tspace.dtype,
            impl=space.tspace.impl)
    if version.parse(odl.__version__) <= version.parse('0.7.0'):
        if not (isinstance(space, odl.discr.DiscreteLp) and
                isinstance(space.fspace.domain, odl.IntervalProd)):
            raise NotImplementedError(
                'for odl <= 0.7.0 only `DiscreteLp` spaces on `IntervalProd` '
                'sets are supported')
        function_space = odl.FunctionSpace(
            odl.IntervalProd([space.fspace.domain.min_pt[0],
                              space.fspace.domain.min_pt[1],
                              z_partition.min_pt[0]],
                             [space.fspace.domain.max_pt[0],
                              space.fspace.domain.max_pt[1],
                              z_partition.max_pt[0]]),
            out_dtype=space.fspace.out_dtype
            )
        space3d = odl.discr.DiscreteLp(
            function_space, rect_partition, tensor_space)
        return space3d
    space3d = odl.discr.DiscretizedSpace(rect_partition, tensor_space)
    return space3d
