# -*- coding: utf-8 -*-

"""SVG support for tomosipo

"""
import tomosipo as ts
import tomosipo.vector_calc as vc
import base64
import uuid
import collections
import numpy as np
from pathlib import Path
from functools import singledispatch
from tomosipo.geometry.base_projection import ProjectionGeometry
from tomosipo.geometry.volume import VolumeGeometry
from tomosipo.geometry.volume_vec import VolumeVectorGeometry

###############################################################################
#                             Showing, saving SVG                             #
###############################################################################


class SVG:
    def __init__(self, svg_str, height=200, width=320, base64=False):
        super().__init__()
        self.svg_str = svg_str
        self.height = height
        self.width = width
        self.base64 = base64

    def _repr_html_(self):
        # How to embed svg:
        # https://vecta.io/blog/best-way-to-embed-svg
        # Display text on hover:
        # https://stackoverflow.com/questions/4697100/accessibility-recommended-alt-text-convention-for-svg-and-mathml
        title = "Click to pause/unpause; press and hold to scroll through animation"

        # Base64 is better when the text is further processed in blogging
        # platforms with markdown etc.. For interactive uses with Jupyter,
        # embedding the svg directly is better since the hover text (title) works.
        if self.base64:
            svg64 = base64.encodebytes(self.svg_str.encode()).decode("ascii")
            tag = r"""<object height="{height}" width="{width}" data="data:image/svg+xml;base64,{image}" />"""
            return tag.format(height=self.height, width=self.width, image=svg64)
        else:
            tag = r"""<object height="{height}" width="{width}" title="{title}"> {svg_str} </object>"""
            return tag.format(
                height=self.height, width=self.width, title=title, svg_str=self.svg_str
            )

    def save(self, path):
        """Save svg to disk"""
        path = Path(path)
        path.write_text(self.svg_str)

    def __str__(self):
        return self.svg_str


###############################################################################
#                               Data structures                               #
###############################################################################
LineItem = collections.namedtuple("LineItem", ["pos", "width", "color"])


def line_item(pos, width=1, color=(0.0, 0.0, 0.0, 1.0)):
    """Create a line item

    :param pos: (N,3) array of floats specifying point locations.
    :param width: width of the line in UNITS TODO
    :param color: tuple of floats in [0.0-1.0] specifying a single color for the entire item in rgba format.
    :returns: a LineItem
    :rtype: LineItem

    """

    pos = np.array(pos, dtype=np.float64, copy=False)
    assert pos.ndim == 2
    assert pos.shape[1] == 3

    assert len(color) == 4

    return LineItem(pos=pos, width=float(width), color=tuple(color))


###############################################################################
#                         From geometries to LineItems                        #
###############################################################################
@singledispatch
def to_line_items(g, i):
    pass


@to_line_items.register(VolumeVectorGeometry)
def vol_vec_to_line_items(vg, i):
    N = len(vg)

    corners = vg.corners[i % N]
    indices = [
        (0, 1),  # <- Bottom square
        (2, 3),  # |
        (4, 5),  # |
        (6, 7),  # |
        (0, 2),  # <- Top square
        (1, 3),  # |
        (4, 6),  # |
        (5, 7),  # |
        (0, 4),  # <- Connecting (vertical) lines
        (1, 5),  # |
        (2, 6),  # |
        (3, 7),  # |
    ]

    return [line_item(pos=np.stack((corners[a], corners[b]))) for a, b in indices]


@to_line_items.register(VolumeGeometry)
def vol_to_line_items(g, i):
    return to_line_items(g.to_vec(), i)


@to_line_items.register(ProjectionGeometry)
def pg_to_line_items(pg, i):
    pg = pg.to_vec()
    i = i % len(pg)

    det_curve = line_item(pos=pg.det_pos, width=0.2)

    # detector plane
    corners = pg.corners[i]
    det_plane_indices = [0, 1, 3, 2, 0]
    det_plane = line_item(pos=corners[det_plane_indices])

    if pg.is_cone:
        src_pos = pg.src_pos[i]
        rays = [line_item(pos=np.stack((src_pos, c)), width=0.2) for c in corners]
        src_curve = line_item(pos=pg.src_pos, width=0.2)
        return [src_curve, det_curve, det_plane, *rays]
    if pg.is_parallel:
        return [det_curve, det_plane]


###############################################################################
#                Projection: From 3D LineItems to 2D LineItems                #
###############################################################################

# Introduction to projection matrices in Python and opengl:
# https://www.labri.fr/perso/nrougier/python-opengl/#id14

# We reuse the cone geometry as a kind of reverse camera: instead of having the
# object located between the source and detector, the source and detector define
# the camera, and the scene is "behind the detector" from the point of view of
# the source. The cone angle determines the field of view.


def default_camera(height, width, angle=1 / 2.7):
    # default camera:
    R0 = ts.rotate(pos=0, axis=(1, 0, 0), deg=70)
    R1 = ts.rotate(pos=0, axis=(0, 0, 1), deg=-25)
    size = (1, 1 * width / height)

    good_cone = (
        R0 * R1 * ts.cone(cone_angle=angle, shape=(height, width), size=size).to_vec()
    )
    camera = ts.translate(good_cone.src_pos * 10 * angle) * good_cone

    return camera


def project_pos(camera, pos):
    assert len(camera) == 1, "Camera must have single viewpoint"

    pos = vc.to_vec(ts.utils.to_pos(pos))
    # Project onto camera coordinates:
    pos = np.array([camera.project_point(p)[0] for p in pos])
    # Move (0, 0) from detector center to detector lower-left corner
    pos += np.array(camera.det_shape)[None, :] / 2

    # Add back z-dimension (equal to zero)
    pos = np.concatenate(
        [np.zeros((len(pos), 1)), pos],
        axis=1,
    )
    return pos


def project_lines(camera, line_items):
    return [
        line_item(
            pos=project_pos(camera, l.pos),
            width=l.width,
            color=l.color,
        )
        for l in line_items
    ]


###############################################################################
#                          From LineItems to SVG text                         #
###############################################################################


def text_svg_frame(
    line_items, frame_begin, frame_end, total_duration, height=100, width=100
):
    # For polylines, see:
    # - https://www.w3.org/TR/SVG11/shapes.html#PolylineElement
    # - https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Basic_Shapes

    polylines = []
    for l in line_items:
        # extract y and x coordinates (from zyx to xy)
        pos2d = l.pos[:, (2, 1)]
        # Correct y to move from bottom to top instead of vice versa
        pos2d[:, 1] = height - pos2d[:, 1]
        points = " ".join(f"{p:0.2f}" for p in pos2d.ravel())
        polyline = f'<polyline points="{points}" stroke="black" fill="none" stroke-width="{l.width:0.2f}"/>'
        polylines.append(polyline)
    lines_str = "\n".join(polylines)

    return f"""<g>
          {lines_str}
          <animate
              attributeName="display"
              values="none;inline;none;none"
              keyTimes="0;{frame_begin};{frame_end};1"
              dur="{total_duration}s"
              begin="0s"
              repeatCount="indefinite" />
     </g>
    """


def text_svg_animation(line_items, duration=10, height=100, width=100):
    frame_duration = 1 / len(line_items)

    const_opts = dict(total_duration=duration, height=height, width=width)

    frames_str = "\n".join(
        text_svg_frame(ls, i * frame_duration, (i + 1) * frame_duration, **const_opts)
        for i, ls in enumerate(line_items)
    )
    # We format this string with `replace` because it is littered with {}'s.

    svg_id = str(uuid.uuid4())

    # Great tutorial on SVG animation:
    # - http://www.joningram.co.uk/article/svg-smil-frame-animation/
    # Docs on onclick and onmousemove:
    # - https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onmousemove
    # - https://stackoverflow.com/questions/16472224/add-onclick-event-to-svg-element
    # - https://stackoverflow.com/questions/10139460/modify-stroke-and-fill-of-svg-image-with-javascript
    # - https://stackoverflow.com/questions/6764961/change-an-image-with-onclick
    # - https://developer.mozilla.org/en-US/docs/Web/API/GlobalEventHandlers/onclick
    # Set current time:
    # - https://www.w3.org/TR/SVG11/struct.html#__svg__SVGSVGElement__setCurrentTime
    # SVG Animations:
    # - https://webplatform.github.io/docs/svg/tutorials/smarter_svg_animation/#Scripting-animations
    # - https://webplatform.github.io/docs/svg/methods/
    # - https://developer.mozilla.org/en-US/docs/Web/API/SVGAnimationElement
    # - https://developer.mozilla.org/en-US/docs/Web/SVG/Element/animate
    # SVG DOM element width:
    # - https://stackoverflow.com/a/18148142
    # JS query selectors (and foreach)
    # - https://developer.mozilla.org/en-US/docs/Web/API/Element/querySelectorAll
    # - https://stackoverflow.com/questions/19324700/how-to-loop-through-all-the-elements-returned-from-getelementsbytagname

    JS = r"""
      <script type="text/ecmascript"><![CDATA[
          function mouse_move(evt) {
              if (evt.buttons > 0) {
                    var root = document.getElementById('SVG_ID');
                    root.pauseAnimations();
                    var new_time = DURATION * evt.clientX / evt.target.getBBox().width;
                    root.setCurrentTime(new_time);
              }
          }
          function on_click(evt) {
              var root = document.getElementById('SVG_ID');
              //console.log(root);
              root.animationsPaused() ? root.unpauseAnimations() : root.pauseAnimations();
          }
        ]]>
      </script>

    """.replace(
        "DURATION", str(duration)
    ).replace(
        "SVG_ID", svg_id
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" id="{svg_id}" height="{height}" width="{width}">
      <title>Click to pause/unpause, press and hold to scroll through animation</title>
        {frames_str}
        {JS}
        <rect x="1" y="1" rx="5" ry="5" onmousemove='mouse_move(evt)' onclick='on_click(evt)' width="{width - 2}" height="{height - 2}" stroke="gray" fill="transparent" stroke-width="1"/>
    </svg>
    """


###############################################################################
#                Tying it all together: from *geometries => svg               #
###############################################################################
def svg(*geoms, height=200, width=320, duration=3, camera=None, base64=False):
    num_steps = max(map(len, geoms))

    # default camera:
    if camera is None:
        c = default_camera(height, width)

    def geoms2frame(i):
        # For each geometry, generate a list of line items
        frames_list = (project_lines(c, to_line_items(g, i)) for g in geoms)
        # flatten the list
        return [f for fs in frames_list for f in fs]

    svg_text = text_svg_animation(
        [geoms2frame(i) for i in range(num_steps)],
        duration=duration,
        height=height,
        width=width,
    )

    return SVG(svg_text, height=height, width=width, base64=base64)
