import numpy as np
import itertools
import tomosipo as ts
from tomosipo.display import get_app, rainbow_colormap
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from .volume import VolumeGeometry, is_volume
from .base_projection import ProjectionGeometry, is_projection


def _pg_items(pg, colors, i):
    pg = pg.to_vec()
    i = i % pg.num_angles
    src_curve_color, det_color, *_ = colors

    # Drawing options:
    curve_opts = dict(width=1, mode="line_strip", antialias=True)
    det_curve_opts = dict(**curve_opts, color=det_color)
    src_curve_opts = dict(**curve_opts, color=src_curve_color)
    ray_opts = dict(**curve_opts, color=(0.0, 0.0, 0.0, .3))
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


def _vg_item(vg, color):
    corners = np.array(vg.get_corners())[:, ::-1]
    volume_mesh = np.array(list(itertools.product(corners, corners, corners)))

    return gl.GLMeshItem(
        vertexes=volume_mesh, smooth=True, color=color, drawEdges=True, drawFaces=True
    )


def _take(xs, n):
    r = []
    for _ in range(n):
        r.append(next(xs))
    return r


@ts.display.register(VolumeGeometry)
@ts.display.register(ProjectionGeometry)
def display_geometry(*geometries):
    """Display a 3D animation of the acquisition geometry

    Note: requires the installation of pyopengl.

    :param geometries:
        Any combination of volume and projection geometries.
    :returns: Nothing
    :rtype: None

    """

    pgs = [g for g in geometries if is_projection(g)]
    vgs = [g for g in geometries if is_volume(g)]

    app = get_app()
    view = gl.GLViewWidget()
    view.setBackgroundColor(0.95)
    view.show()

    idx = []
    for i in range(8):
        idx = idx + list(range(i, 256, 32))
    colors = map(tuple, rainbow_colormap[idx])

    for vg in vgs:
        color, *colors = colors
        view.addItem(_vg_item(vg, color))

    colors = itertools.cycle(colors)
    pg_colors = [tuple(_take(colors, 2)) for _ in pgs]

    print(pg_colors)
    assert len(pg_colors) == len(pgs)

    tmp_items = []
    i = 0

    def on_timer():
        nonlocal i, tmp_items, pg_colors
        for item in tmp_items:
            view.removeItem(item)
        tmp_items = []
        for pg, c in zip(pgs, pg_colors):
            for item in _pg_items(pg, c, i):
                view.addItem(item)
                tmp_items.append(item)
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(on_timer)
    max_angles = max(pg.num_angles for pg in pgs)
    timer.start(5000 / max_angles)
    on_timer()
    app.exec_()
