import numpy as np
import itertools
import tomosipo as ts
from tomosipo.display import get_app, rainbow_colormap
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from .volume import VolumeGeometry, is_volume
from .base_projection import ProjectionGeometry, is_projection


@ts.display.register(VolumeGeometry)
@ts.display.register(ProjectionGeometry)
def display_geometry(*geometries):
    """Display a 3D animation of the acquisition geometry

    Note: requires the installation of pyopengl.

    :param pg: ProjectionGeometry
        Only cone beam geometries are supported currently.
    :param vg: VolumeGeometry
    :returns: Nothing
    :rtype: None

    """

    pgs = [g for g in geometries if is_projection(g)]
    vgs = [g for g in geometries if is_volume(g)]

    if len(pgs) != 1:
        raise ValueError(
            f"Expected 1 projection geometry to be supplied to ts.display_geometry. Got: {len(pgs)}"
        )

    if len(vgs) != 1:
        raise ValueError(
            f"Expected 1 volume geometry to be supplied to ts.display_geometry. Got: {len(vgs)}"
        )

    pg, *_ = pgs
    vg, *_ = vgs

    app = get_app()
    view = gl.GLViewWidget()
    # view.setBackgroundColor('w')

    view.show()

    pg = pg.to_vec()

    # Get source positions in X, Y, Z order
    if pg.is_cone:
        source_pos = pg.src_pos[:, ::-1]
    if pg.is_parallel:
        ray_dir = pg.ray_dir[:, ::-1]  # XYZ order
    # Get detector corners shaped (num_angles, 4 -- for each corner,
    # 3) and in XYZ order.
    detector_corners = pg.corners[:, :, ::-1]
    # Calculate detector center from the corners:
    detector_origins = np.mean(detector_corners, axis=1)
    #######################################################################
    #                          Draw source curve                          #
    #######################################################################
    if pg.is_cone:
        sourceItem = gl.GLLinePlotItem(pos=source_pos, width=1, mode="line_strip")
        view.addItem(sourceItem)

    #######################################################################
    #                         Draw detector curve                         #
    #######################################################################
    # Detector positions in (X, Y, Z) formation

    detector_curve = gl.GLLinePlotItem(
        pos=detector_origins, width=1, color=(0, 1, 0, 1), mode="line_strip"
    )
    view.addItem(detector_curve)

    #######################################################################
    #                         Show volume geometry                        #
    #######################################################################
    # 8 corners in (XYZ) formation
    corners = np.array(vg.get_corners())[:, ::-1]
    volume_mesh = np.array(list(itertools.product(corners, corners, corners)))

    volumeItem = gl.GLMeshItem(
        vertexes=volume_mesh,
        smooth=False,
        color=(0.0, 1.0, 1.0, 1.0),
        drawEdges=False,
        drawFaces=True,
    )
    view.addItem(volumeItem)

    #######################################################################
    #                     Draw detector at first angle                    #
    #######################################################################
    # Show detector position i
    def draw_detector(i):
        i = i % len(detector_origins)

        corners = detector_corners[i]
        meshdata = np.array(list(itertools.product(corners, repeat=3)))

        detectorItem = gl.GLMeshItem(
            vertexes=meshdata, smooth=False, color=(1.0, 0, 0, 0.9), drawEdges=False
        )
        return detectorItem

    #######################################################################
    #                  Draw rays through detector corners                 #
    #######################################################################

    def draw_corner_rays(i):
        i = i % len(detector_origins)

        corners = detector_corners[i]
        if pg.is_cone:
            source = np.copy(source_pos[i])
            rayItems = [
                gl.GLLinePlotItem(
                    pos=np.array([source, corner]), width=1, mode="line_strip"
                )
                for corner in corners
            ]
        elif pg.is_parallel:
            ray = np.copy(ray_dir[i])
            ray *= 10 * np.linalg.norm(pg.det_sizes[0])
            rayItems = [
                gl.GLLinePlotItem(
                    pos=np.array([corner - 10 * ray, corner + 10 * ray]),
                    width=1,
                    mode="line_strip",
                )
                for corner in corners
            ]

        return rayItems

    ray_items = None
    detector_item = None
    i = 0

    def on_timer():
        nonlocal ray_items, detector_item, i
        if ray_items is not None:
            for item in ray_items:
                view.removeItem(item)
            ray_items = None
        if detector_item is not None:
            view.removeItem(detector_item)
            detector_item = None
        ray_items = draw_corner_rays(i)
        detector_item = draw_detector(i)
        view.addItem(detector_item)
        for r in ray_items:
            view.addItem(r)
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(on_timer)
    timer.start(5000 / len(detector_origins))
    on_timer()
    app.exec_()
