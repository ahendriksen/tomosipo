""" This module contains a modified version of GLViewWidget.renderToArray().

The modification is described in this pull request:

https://github.com/pyqtgraph/pyqtgraph/pull/1306

At some point, we should consider removing this function.

"""
from OpenGL.GL import *
import OpenGL.GL.framebufferobjects as glfbo
import numpy as np


def render_to_array(gl_view_widget, shape, textureSize=1024, padding=256):
    format = GL_RGBA
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
                    np.frombuffer(data, dtype=np.ubyte)
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
