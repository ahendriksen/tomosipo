import itertools
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from .display import (
    get_app,
    rainbow_colormap,
    display_backends,
)
from tomosipo.geometry.base_projection import ProjectionGeometry, is_projection
from tomosipo.geometry.volume import VolumeGeometry, is_volume
from tomosipo.geometry.volume_vec import VolumeVectorGeometry
from functools import singledispatch


@singledispatch
def to_mesh_items(g, colors, i):
    pass


@singledispatch
def num_colors(g):
    pass


@num_colors.register(VolumeGeometry)
def vg_num_colors(g):
    return 1


@num_colors.register(VolumeVectorGeometry)
def vg_vec_num_colors(g):
    return 1


@num_colors.register(ProjectionGeometry)
def pg_num_colors(g):
    return 2


@to_mesh_items.register(ProjectionGeometry)
def pg_to_mesh_items(pg, colors, i):
    pg = pg.to_vec()
    i = i % pg.num_angles
    src_curve_color, det_color, *_ = colors

    # Drawing options:
    curve_opts = dict(width=1, mode="line_strip", antialias=True)
    det_curve_opts = dict(**curve_opts, color=det_color)
    src_curve_opts = dict(**curve_opts, color=src_curve_color)
    ray_opts = dict(**curve_opts, color=(0.0, 0.0, 0.0, 0.3))
    det_opts = dict(smooth=True, drawEdges=True, color=det_color)

    # Detector Curve
    det_curve = gl.GLLinePlotItem(pos=pg.det_pos[:, ::-1], **det_curve_opts)

    # Detector
    # Get corners in XYZ coordinates:
    corners = pg.corners[i, :, ::-1]
    meshdata = np.array(list(itertools.product(corners, repeat=3)))
    det_item = gl.GLMeshItem(vertexes=meshdata, **det_opts)

    if pg.is_cone:
        # Source curve
        src_item = gl.GLLinePlotItem(pos=pg.src_pos[:, ::-1], **src_curve_opts)
        # Rays
        source = pg.src_pos[i, ::-1]
        ray_items = [
            gl.GLLinePlotItem(pos=np.array([source, corner]), **ray_opts)
            for corner in corners
        ]
        return (src_item, *ray_items, det_curve, det_item)
    elif pg.is_parallel:
        # Rays
        ray = pg.ray_dir[i, ::-1]  # XYZ order
        ray *= 100 * np.linalg.norm(pg.det_sizes[0])
        ray_items = [
            gl.GLLinePlotItem(pos=np.array([corner - ray, corner + ray]), **ray_opts)
            for corner in corners
        ]
        return (*ray_items, det_curve, det_item)

    raise ValueError("Expected cone or parallel geometry")


@to_mesh_items.register(VolumeGeometry)
def vg_to_mesh_items(vg, colors, i):
    return vg_vec_to_mesh_items(vg.to_vec(), colors, i)


@to_mesh_items.register(VolumeVectorGeometry)
def vg_vec_to_mesh_items(vg, colors, i):
    color = colors[0]
    i = i % vg.num_steps
    c = vg.corners[i, :, ::-1]
    volume_mesh = np.array(list(itertools.product(c, c, c)))
    return [
        gl.GLMeshItem(
            vertexes=volume_mesh,
            smooth=False,
            color=color,
            drawEdges=True,
            drawFaces=True,
        )
    ]


def _take(xs, n):
    r = []
    for _ in range(n):
        r.append(next(xs))
    return tuple(r)
