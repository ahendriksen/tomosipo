import numpy as np
from .utils import up_tuple
import tomosipo as ts
import pyqtgraph as pq
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from tomosipo.display import run_app, get_app
import itertools
from . import vector_calc as vc


class OrientedBox(object):
    """Documentation for OrientedBox

    """

    def __init__(self, size, pos, w=(1, 0, 0), v=(0, 1, 0), u=(0, 0, 1)):
        """Create a new oriented box

        An oriented box with multiple orientations and positions.

        The position describes the center of the box.

        :param size: `(scalar, scalar, scalar)` or `scalar`
            The size of the box as measured in basis elements w,
            v, u.
        :param pos: `scalar`, `np.array`
            A numpy array of dimension (num_orientations, 3)
            describing the position of the box in world-coordinates
            `(Z, Y, X)`. You may also pass a 3-tuple or a scalar.
        :param w: `np.array` (optional)
            A numpy array of dimension (num_orientations, 3)
            describing the `w` basis element in `(Z, Y, X)`
            coordinates. Default is `(1, 0, 0)`.
        :param v: `np.array` (optional)
            A numpy array of dimension (num_orientations, 3)
            describing the `v` basis element in `(Z, Y, X)`
            coordinates. Default is `(0, 1, 0)`.
        :param u: `np.array` (optional)
            A numpy array of dimension (num_orientations, 3)
            describing the `u` basis element in `(Z, Y, X)`
            coordinates. Default is `(0, 0, 1)`.
        :returns:
        :rtype:

        """
        super(OrientedBox, self).__init__()

        self.size = up_tuple(size, 3)
        if np.isscalar(pos):
            pos = up_tuple(pos, 3)

        pos, w, v, u = np.broadcast_arrays(*(vc.to_vec(x) for x in (pos, w, v, u)))
        self.pos, self.w, self.v, self.u = pos, w, v, u

        shapes = [x.shape for x in [self.pos, self.w, self.v, self.u]]

        if min(shapes) != max(shapes):
            raise ValueError(
                "Not all arguments pos, w, v, u are the same shape. " f"Got: {shapes}"
            )

    def __repr__(self):
        return (
            f"OrientedBox(\n"
            f"    size={self.size},\n"
            f"    pos={self.pos},\n"
            f"    w={self.w},\n"
            f"    v={self.v},\n"
            f"    u={self.u}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, OrientedBox):
            return False

        d_pos = self.pos - other.pos
        d_w = self.size[0] * self.w - other.size[0] * other.w
        d_v = self.size[1] * self.v - other.size[1] * other.v
        d_u = self.size[2] * self.u - other.size[2] * other.u

        return (
            np.all(abs(d_pos) < ts.epsilon)
            and np.all(abs(d_w) < ts.epsilon)
            and np.all(abs(d_v) < ts.epsilon)
            and np.all(abs(d_u) < ts.epsilon)
        )

    @property
    def corners(self):
        c = np.array(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        )
        c = c - 0.5

        size = self.size
        c_w = c[:, 0:1, None] * self.w * size[0]
        c_v = c[:, 1:2, None] * self.v * size[1]
        c_u = c[:, 2:3, None] * self.u * size[2]

        c = self.pos + c_w + c_v + c_u

        return c.swapaxes(0, 1)

    @property
    def num_orientations(self):
        return len(self.pos)

    def transform(self, matrix):
        pos = vc.to_homogeneous_point(self.pos)
        w = vc.to_homogeneous_vec(self.w)
        v = vc.to_homogeneous_vec(self.v)
        u = vc.to_homogeneous_vec(self.u)

        new_pos = vc.to_vec(vc.matrix_transform(matrix, pos))
        new_w = vc.to_vec(vc.matrix_transform(matrix, w))
        new_v = vc.to_vec(vc.matrix_transform(matrix, v))
        new_u = vc.to_vec(vc.matrix_transform(matrix, u))

        return OrientedBox(self.size, new_pos, new_w, new_v, new_u)


@ts.display.register(OrientedBox)
def display_oriented_box(*boxes):
    app = get_app()
    view = gl.GLViewWidget()
    view.show()

    box, *_ = boxes

    #######################################################################
    #                         Show volume geometry                        #
    #######################################################################

    def draw_orientation(box, i, color=(0.0, 1.0, 1.0, 1.0)):
        # 8 corners in (XYZ) formation
        i = i % box.num_orientations
        c = box.corners[i, :, ::-1]
        volume_mesh = np.array(list(itertools.product(c, c, c)))
        return gl.GLMeshItem(
            vertexes=volume_mesh,
            smooth=False,
            color=(1.0, 1.0, 1.0, 1.0),
            drawEdges=True,
            drawFaces=True,
        )

    i = 0
    meshes = []

    def on_timer():
        nonlocal boxes, i, meshes

        # Remove old meshes:
        for m in meshes:
            view.removeItem(m)

        meshes = []
        for box in boxes:
            m = draw_orientation(box, i)
            meshes.append(m)
            view.addItem(m)
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(on_timer)
    timer.start(5000 / box.num_orientations)
    run_app(app)
