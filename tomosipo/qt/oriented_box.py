import itertools
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from .display import (
    get_app,
    rainbow_colormap,
    run_app,
    display_backends,
)
from tomosipo.geometry.oriented_box import OrientedBox


def _box_item(box, i, color=(1.0, 1.0, 1.0, 1.0)):
    # 8 corners in (XYZ) formation
    i = i % box.num_steps
    c = box.corners[i, :, ::-1]
    volume_mesh = np.array(list(itertools.product(c, c, c)))
    return gl.GLMeshItem(
        vertexes=volume_mesh, smooth=False, color=color, drawEdges=True, drawFaces=True,
    )


def display_oriented_box(*boxes):
    app = get_app()
    view = gl.GLViewWidget()
    view.setBackgroundColor(0.95)
    view.show()

    idx = []
    for i in range(16):
        idx = idx + list(range(i, 256, 16))

    colors = rainbow_colormap[idx]

    #######################################################################
    #                         Show volume geometry                        #
    #######################################################################

    i = 0
    meshes = []

    def on_timer():
        nonlocal boxes, i, meshes

        # Remove old meshes:
        for m in meshes:
            view.removeItem(m)

        meshes = []
        for (box, color) in zip(boxes, colors):
            m = _box_item(box, i, color=color)
            meshes.append(m)
            view.addItem(m)
        i += 1

    view.setCameraPosition(
        pos=boxes[0].pos, distance=5 * np.sqrt(sum(np.square(boxes[0].abs_size[0])))
    )
    timer = QtCore.QTimer()
    timer.timeout.connect(on_timer)
    max_orientations = max(b.num_steps for b in boxes)
    timer.start(5000 / max_orientations)
    on_timer()
    run_app(app)


display_backends[OrientedBox] = display_oriented_box
