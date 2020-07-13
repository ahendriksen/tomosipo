# First, import from tomosipo.qt to catch import errors.
from tomosipo.qt.geometry import (
    _vg_item,
    _pg_items,
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

from tomosipo.geometry.base_projection import is_projection
from tomosipo.geometry.volume import is_volume

import matplotlib.pyplot as plt
from matplotlib import animation
import warnings

if not animation.writers.is_available('ffmpeg'):
    warnings.warn(
        "FFMPEG encoding is not available. Make sure that ffmpeg is installed using: \n"
        "> conda install ffmpeg \n"
        "In addition use \n"
        "> import matplotlib.pyplot as plt \n"
        "> plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg' \n"
        "before importing `tomosipo.qt.animate'. \n"
    )


def geometry_animation(*geometries, total_length_seconds=3):
    if plt.rcParams['animation.html'] != 'html5':
        warnings.warn(
            "For proper animated videos in Jupyter notebooks, make sure that you set \n"
            "> plt.rc('animation', html='html5') \n"
        )
    video = geometry_video(*geometries)

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ims = ax.imshow(video[0])
    fig.tight_layout()

    def animate(i):
        nonlocal ims
        ims = ax.imshow(video[i])
        return (ims,)

    interval_ms = 1000 * total_length_seconds / len(video)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(video),
        interval=interval_ms,
        blit=False
    )
    return anim


def geometry_video(*geometries):
    """Display a 3D animation of the acquisition geometry

    :param geometries:
        Any combination of volume and projection geometries.
    :returns: A 4-dimensional numpy array (Time, H, W, C).
    :rtype: None

    """

    pgs = [g for g in geometries if is_projection(g)]
    vgs = [g for g in geometries if is_volume(g)]

    _ = get_app()
    view = gl.GLViewWidget()
    view.setBackgroundColor(0.95)

    idx = []
    for i in range(8):
        idx = idx + list(range(i, 256, 32))
    colors = map(tuple, rainbow_colormap[idx])

    for vg in vgs:
        color, *colors = colors
        view.addItem(_vg_item(vg, color))

    colors = itertools.cycle(colors)
    pg_colors = [tuple(_take(colors, 2)) for _ in pgs]

    assert len(pg_colors) == len(pgs)

    tmp_items = []

    max_angles = max([1, *(pg.num_angles for pg in pgs)])
    shape = (640, 480)
    out_video = np.zeros((max_angles, shape[1], shape[0], 4), dtype=np.uint8)

    for i in range(max_angles):
        for item in tmp_items:
            view.removeItem(item)
        tmp_items = []
        for pg, c in zip(pgs, pg_colors):
            for item in _pg_items(pg, c, i):
                view.addItem(item)
                tmp_items.append(item)

        video_frame = render_to_array(view, shape)
        # Flip x, y
        video_frame = video_frame.transpose(1, 0, 2)
        # Move alpha channel to back (ARGB -> RGBA)
        video_frame = np.roll(video_frame, 3, axis=2)
        out_video[i] = video_frame

    return out_video


def render_to_array(gl_view_widget, size, textureSize=1024, padding=256):
    # This is a modified version of GLViewWidget.renderToArray().
    # Here, this PR has been incorporated:
    # https://github.com/pyqtgraph/pyqtgraph/pull/1306
    # At some point, we should consider removing this function.
    format = GL_BGRA
    type = GL_UNSIGNED_BYTE
    w, h = map(int, size)

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
        glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])
        # Create texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.transpose((1,0,2)))

        # Create depth buffer
        depth_buf = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, texwidth, texwidth)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buf)

        gl_view_widget.opts['viewport'] = (0, 0, w, h)  # viewport is the complete image; this ensures that paintGL(region=...)
                                              # is interpreted correctly.
        p2 = 2 * padding
        for x in range(-padding, w-padding, texwidth-p2):
            for y in range(-padding, h-padding, texwidth-p2):
                x2 = min(x+texwidth, w+padding)
                y2 = min(y+texwidth, h+padding)
                w2 = x2-x
                h2 = y2-y

                ## render to texture
                glfbo.glFramebufferTexture2D(glfbo.GL_FRAMEBUFFER, glfbo.GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

                gl_view_widget.paintGL(region=(x, h-y-h2, w2, h2), viewport=(0, 0, w2, h2))  # only render sub-region
                glBindTexture(GL_TEXTURE_2D, tex) # fixes issue #366

                ## read texture back to array
                data = glGetTexImage(GL_TEXTURE_2D, 0, format, type)
                data = np.fromstring(data, dtype=np.ubyte).reshape(texwidth,texwidth,4).transpose(1,0,2)[:, ::-1]
                output[x+padding:x2-padding, y+padding:y2-padding] = data[padding:w2-padding, -(h2-padding):-padding]

    finally:
        gl_view_widget.opts['viewport'] = None
        glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        if tex is not None:
            glDeleteTextures([tex])
        if fb is not None:
            glfbo.glDeleteFramebuffers([fb])
        if depth_buf is not None:
            glDeleteRenderbuffers(1, [depth_buf])

    return output
