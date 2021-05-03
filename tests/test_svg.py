#!/usr/bin/env python3

import tomosipo as ts
import numpy as np


def test_svg():
    geoms = [
        ts.cone(angles=10, cone_angle=1 / 2).to_vec(),
        ts.parallel(angles=3).to_vec(),
        ts.volume().to_vec(),
    ]

    options = [
        dict(height=100, width=100, duration=3, camera=None, show_axes=True),
        dict(
            height=100,
            width=100,
            duration=3,
            camera=ts.cone(shape=100, cone_angle=1 / 2),
            show_axes=True,
        ),
        dict(height=20, width=20, duration=3, camera=None, show_axes=False),
        dict(height=20, width=20, duration=1.0, camera=None, show_axes=False),
    ]
    for opt in options:
        svg_item = ts.svg(*geoms, **opt)
        svg_item._repr_markdown_()
        svg_item._repr_html_()
        svg_item._repr_svg_()


def test_svg_id_does_not_change():
    geoms = [
        ts.cone(angles=10, cone_angle=1 / 2).to_vec(),
        ts.parallel(angles=3).to_vec(),
        ts.volume().to_vec(),
    ]
    repr1 = str(ts.svg(*geoms))
    repr2 = str(ts.svg(*geoms))
    assert repr1 == repr2
