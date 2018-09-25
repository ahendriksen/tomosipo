import numpy as np
import tomosipo as ts
import itertools
import astra
import warnings
import pyqtgraph as pq
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl


def get_app():
    return pq.mkQApp()


def run_app(app):
    app.exec_()


def display_data(data):
    """Display a projection or volume data set.

    Shows the slices or projection images depending on the argument.

    Note: for projection datasets, the axes on the data are
    swapped. This can induce more allocations. Make sure not to call
    this procedure with very large projection datasets or if memory is
    scarce.

    :param data: `Data`
        A tomosipo dataset of either a volume or projection set.
    :returns: None
    :rtype:

    """

    if data.is_volume():
        app = get_app()
        pq.image(data.get())
        run_app(app)
    elif data.is_projection():
        app = get_app()
        pq.image(data.get().swapaxes(0, 1).swapaxes(1, 2))
        run_app(app)


# TODO: Add pyopengl as dependency
def display_geometry(pg, vg):
    """Display a 3D animation of the acquisition geometry

    Note: requires the installation of pyopengl.

    :param pg: ProjectionGeometry
        Only cone beam geometries are supported currently.
    :param vg: VolumeGeometry
    :returns: Nothing
    :rtype: None

    """
    app = get_app()
    view = gl.GLViewWidget()
    view.show()

    if not pg.is_cone():
        raise NotImplementedError(
            "Displaying of parallel geometries is not yet supported."
        )

    pg = pg.to_vector()

    # Get source positions in X, Y, Z order
    source_pos = pg.get_source_positions()[:, ::-1]
    # Get detector corners shaped (num_angles, 4 -- for each corner,
    # 3) and in XYZ order.
    detector_corners = pg.get_corners()[:, :, ::-1].swapaxes(0, 1)
    # Calculate detector center from the corners:
    detector_origins = np.mean(detector_corners, axis=1)
    #######################################################################
    #                          Draw source curve                          #
    #######################################################################
    if pg.is_cone():
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
        i = i % len(source_pos)

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
        i = i % len(source_pos)
        if pg.is_cone():
            source = np.copy(source_pos[i])
        elif pg.is_parallel():
            # TODO: unimplemented
            pass

        corners = detector_corners[i]
        if pg.is_cone():
            rayItems = [
                gl.GLLinePlotItem(
                    pos=np.array([source, corner]), width=1, mode="line_strip"
                )
                for corner in corners
            ]
        elif pg.is_parallel():
            # TODO: fix this to make parallel display work
            rayItems = [
                gl.GLLinePlotItem(
                    pos=np.array([corner - 1000 * ray_dir, corner + 1000 * ray_dir]),
                    width=1,
                    mode="line_strip",
                )
                for corner in detector_corners
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
    timer.start(5000 / len(source_pos))

    app.exec_()
