#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for astra conversions."""

import pytest
import numpy as np
import tomosipo as ts
from tomosipo.geometry import (
    random_cone,
    random_cone_vec,
    random_parallel,
    random_parallel_vec,
    random_volume,
)
from tomosipo.astra import (forward, backward, fdk)
from . import skip_if_no_cuda

###############################################################################
#                           Test geometry conversion                          #
###############################################################################

def test_to_from_astra():
    rs = [
        random_cone,
        random_cone_vec,
        random_parallel,
        random_parallel_vec,
        random_volume,
    ]

    for r in rs:
        g = r()
        g_astra = ts.to_astra(g)
        g_ts = ts.from_astra(g_astra)
        assert g == g_ts
        with pytest.raises(TypeError):
            ts.from_astra(g)
        with pytest.raises(TypeError):
            ts.to_astra(g_astra)


###############################################################################
#                       Test legacy projection functions                      #
###############################################################################
@skip_if_no_cuda
def test_forward_backward():
    pd = ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2, shape=10))
    vd = ts.data(ts.volume(shape=10))

    rs = [
        ([pd, vd], {}),
        ([pd, vd], dict(detector_supersampling=2, voxel_supersampling=2)),
        ((pd, vd), dict(detector_supersampling=1, voxel_supersampling=2)),
        ((pd, vd), dict(detector_supersampling=2, voxel_supersampling=1)),
    ]

    for data, kwargs in rs:
        forward(*data, **kwargs)
        backward(*data, **kwargs)


@skip_if_no_cuda
def test_fdk(interactive):
    if interactive:
        from tomosipo.qt import display

    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=100, shape=100)
    vg = ts.volume(shape=100)
    pd = ts.data(pg)
    vd = ts.data(vg)

    ts.phantom.hollow_box(vd)
    forward(vd, pd)

    if interactive:
        display(vg, pg)
        display(pd)

    fdk(vd, pd)

    if interactive:
        display(vd)
