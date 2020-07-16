# First, import from tomosipo.qt to catch import errors.
import tomosipo as ts
from tomosipo.qt.geometry import (
    to_mesh_items,
    num_colors,
    _take,
)
from tomosipo.qt.display import (
    get_app,
    rainbow_colormap,
)
import numpy as np
import itertools
from OpenGL.GL import *
import OpenGL.GL.framebufferobjects as glfbo
import pyqtgraph.opengl as gl
import warnings

# TODO: Check that ffmpeg is available.
# warnings.warn(
#     "FFMPEG encoding is not available. Make sure that ffmpeg is installed using: \n"
#     "> conda install ffmpeg \n"
#     "In addition use \n"
#     "> import matplotlib.pyplot as plt \n"
#     "> plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg' \n"
#     "before importing `tomosipo.qt.animate'. \n"
# )

import ffmpeg
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import base64


class Animation(object):
    """Documentation for Animation

    """

    def __init__(self, *geometries):
        super().__init__()
        self.geometries = geometries

    def video_array(self):
        return geometry_video(*self.geometries)

    def save(self, path):
        video = self.video_array()
        height, width = video.shape[1:3]

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
            for frame in video:
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
                path = Path(tmpdir, "temp.m4v")
                path = Path("temp.m4v")

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


def geometry_video(*geometries):
    """Display a 3D animation of the acquisition geometry

    :param geometries:
        Any combination of volume and projection geometries.
    :returns: A 4-dimensional numpy array (Time, H, W, C).
    :rtype: None

    """
    # Video will be 640 pixels wide and 480 high.
    SHAPE = (480, 640)

    # We get the current active instance of the QT app. We do not have
    # to close it.
    _ = get_app()
    view = gl.GLViewWidget()
    view.setBackgroundColor(0.95)

    idx = []
    for i in range(8):
        idx = idx + list(range(i, 256, 32))
    colors = map(tuple, rainbow_colormap[idx])
    colors = itertools.cycle(colors)

    geometry_colors = [_take(colors, num_colors(g)) for g in geometries]

    assert len(geometry_colors) == len(geometries)

    tmp_items = []
    max_steps = max([1, *(g.num_steps for g in geometries)])
    out_video = np.zeros((max_steps, *SHAPE, 4), dtype=np.uint8)

    for i in range(max_steps):
        for item in tmp_items:
            view.removeItem(item)
        tmp_items = []
        for g, c in zip(geometries, geometry_colors):
            for item in to_mesh_items(g, c, i):
                view.addItem(item)
                tmp_items.append(item)

        video_frame = render_to_array(view, SHAPE)
        # Flip x, y
        video_frame = video_frame.transpose(1, 0, 2)
        # Move alpha channel to back (ARGB -> RGBA)
        video_frame = np.roll(video_frame, 3, axis=2)
        out_video[i] = video_frame

    # Close view widget
    view.close()

    return out_video


def render_to_array(gl_view_widget, shape, textureSize=1024, padding=256):
    # This is a modified version of GLViewWidget.renderToArray().
    # Here, this PR has been incorporated:
    # https://github.com/pyqtgraph/pyqtgraph/pull/1306
    # At some point, we should consider removing this function.
    format = GL_BGRA
    type = GL_UNSIGNED_BYTE
    h, w = map(int, shape)

    gl_view_widget.makeCurrent()
    tex = None
    fb = None
    depth_buf = None
    try:
        output = np.empty((w, h, 4), dtype=np.ubyte)
        fb = glfbo.glGenFramebuffers(1)
        glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fb)

        glEnable(GL_TEXTURE_2D)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        texwidth = textureSize
        data = np.zeros((texwidth, texwidth, 4), dtype=np.ubyte)

        # Test texture dimensions first
        glTexImage2D(
            GL_PROXY_TEXTURE_2D,
            0,
            GL_RGBA,
            texwidth,
            texwidth,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            None,
        )
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception(
                "OpenGL failed to create 2D texture (%dx%d); too large for this hardware."
                % shape[:2]
            )
        # Create texture
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            texwidth,
            texwidth,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data.transpose((1, 0, 2)),
        )

        # Create depth buffer
        depth_buf = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, texwidth, texwidth)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buf
        )

        gl_view_widget.opts["viewport"] = (
            0,
            0,
            w,
            h,
        )  # viewport is the complete image; this ensures that paintGL(region=...)
        # is interpreted correctly.
        p2 = 2 * padding
        for x in range(-padding, w - padding, texwidth - p2):
            for y in range(-padding, h - padding, texwidth - p2):
                x2 = min(x + texwidth, w + padding)
                y2 = min(y + texwidth, h + padding)
                w2 = x2 - x
                h2 = y2 - y

                ## render to texture
                glfbo.glFramebufferTexture2D(
                    glfbo.GL_FRAMEBUFFER,
                    glfbo.GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    tex,
                    0,
                )

                gl_view_widget.paintGL(
                    region=(x, h - y - h2, w2, h2), viewport=(0, 0, w2, h2)
                )  # only render sub-region
                glBindTexture(GL_TEXTURE_2D, tex)  # fixes issue #366

                ## read texture back to array
                data = glGetTexImage(GL_TEXTURE_2D, 0, format, type)
                data = (
                    np.fromstring(data, dtype=np.ubyte)
                    .reshape(texwidth, texwidth, 4)
                    .transpose(1, 0, 2)[:, ::-1]
                )
                output[x + padding : x2 - padding, y + padding : y2 - padding] = data[
                    padding : w2 - padding, -(h2 - padding) : -padding
                ]

    finally:
        gl_view_widget.opts["viewport"] = None
        glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        if tex is not None:
            glDeleteTextures([tex])
        if fb is not None:
            glfbo.glDeleteFramebuffers([fb])
        if depth_buf is not None:
            glDeleteRenderbuffers(1, [depth_buf])

    return output
