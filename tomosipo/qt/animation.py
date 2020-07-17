from pathlib import Path
from tempfile import TemporaryDirectory
import base64
import ffmpeg
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import subprocess
from .view_widget_hack import render_to_array
from tomosipo.qt.geometry import (
    to_mesh_items,
    num_colors,
    _take,
)
from tomosipo.qt.display import (
    get_app,
    get_color_cycle,
    display_backends,
)
from tomosipo.geometry import (
    ProjectionGeometry,
    VolumeGeometry,
    VolumeVectorGeometry,
)


def animate(*geometries, total_duration_seconds=5):
    """Returns an animation of the geometries

    The return type is an object that can be saved:

    >>> ts.animate(ts.parallel(angles=10)).save('geometry.mp4')

    It can also be opened in a window:

    >>> ts.animate(ts.parallel(angles=10)).window()

    In a Jupyter notebook, the animation is automatically converted to
    a video and shown if it is the final value in a cell.

    :param total_duration_seconds:
        the total duration of the animation in seconds.
    :returns: an animation object
    :rtype: `Animation`

    """
    return Animation(*geometries, total_duration_seconds=total_duration_seconds)


class Animation(object):
    """Documentation for Animation

    """

    def __init__(self, *geometries, total_duration_seconds=5):
        super().__init__()
        self.VIDEO_SHAPE = (480, 640)
        self.total_duration_seconds = total_duration_seconds
        self.geometries = geometries

    def new_gl_view_widget(self):
        # We get the current active instance of the QT app. We do not have
        # to close it.
        _ = get_app()
        view = gl.GLViewWidget()
        view.setBackgroundColor(0.95)
        return view

    @property
    def total_steps(self):
        return max([1, *(g.num_steps for g in self.geometries)])

    def iter_video_frames(self):
        """Return an iterator over the frames the video
        """
        color_cycle = get_color_cycle()
        colors = [_take(color_cycle, num_colors(g)) for g in self.geometries]

        widget = self.new_gl_view_widget()
        for step in range(self.total_steps):
            update_widget(widget, self.geometries, colors, step)
            video_frame = render_to_array(widget, self.VIDEO_SHAPE)
            # Flip x, y
            video_frame = video_frame.transpose(1, 0, 2)
            yield video_frame

        # Close view widget
        widget.close()

    def video_as_array(self):
        out_video = np.zeros((self.total_steps, *self.VIDEO_SHAPE, 4), dtype=np.uint8)
        for i, frame in enumerate(self.iter_video_frames()):
            out_video[i] = frame
        return out_video

    def window(self):
        app = get_app()
        widget = self.new_gl_view_widget()
        widget.show()
        color_cycle = get_color_cycle()
        colors = [_take(color_cycle, num_colors(g)) for g in self.geometries]
        i = 0

        def on_timer():
            nonlocal i
            update_widget(widget, self.geometries, colors, i)
            i += 1

        timer = QtCore.QTimer()
        timer.timeout.connect(on_timer)
        # The timer interval is in milliseconds
        timer.start(self.total_duration_seconds * 1000 / self.total_steps)
        on_timer()
        app.exec_()

    def save(self, path="geometry_video.mp4"):
        height, width = self.VIDEO_SHAPE

        args = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
            )
            .output(str(path), vcodec="h264", pix_fmt="yuv420p")
            .overwrite_output()
            .compile()
        )
        process = None
        try:
            process = subprocess.Popen(args, stdin=subprocess.PIPE)
            for frame in self.iter_video_frames():
                frame = frame[..., :3]  # Remove alpha channel
                process.stdin.write(frame.astype(np.uint8).tobytes())
        finally:
            if process is not None:
                process.stdin.close()
                process.wait()

    def _repr_html_(self):
        if hasattr(self, "base64_video"):
            return self.base64_video
        else:
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "geometry_video.m4v")
                self.save(path)
                # Now open and base64 encode.
                vid64 = base64.encodebytes(path.read_bytes()).decode("ascii")

                options = ["controls", "autoplay", "loop"]
                VIDEO_TAG = r"""<video width="320" height="240" {options}>
              <source type="video/mp4" src="data:video/mp4;base64,{video}">
              Your browser does not support the video tag.
            </video>"""
                self.base64_video = VIDEO_TAG.format(
                    video=vid64, options=" ".join(options)
                )

                return self.base64_video


def update_widget(widget, geometries, colors, step):
    # Clear widget
    while len(widget.items) > 0:
        widget.removeItem(widget.items[0])
    # Add items for current step:
    for g, c in zip(geometries, colors):
        for item in to_mesh_items(g, c, step):
            widget.addItem(item)


def display_geometry(*geometries):
    """Display interactive 3D animation of the acquisition geometry

    Note: requires the installation of pyopengl.

    :param geometries:
        Any combination of volume and projection geometries.
    :returns: Nothing
    :rtype: None

    """
    animation = Animation(*geometries)
    animation.window()


display_backends[ProjectionGeometry] = display_geometry
display_backends[VolumeGeometry] = display_geometry
display_backends[VolumeVectorGeometry] = display_geometry
