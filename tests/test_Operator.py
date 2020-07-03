#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for operator."""

from pytest import approx
import numpy as np
import tomosipo as ts


def test_forward_backward():
    pd = ts.data(ts.cone(shape=10))
    vd = ts.data(ts.volume(10))

    rs = [
        ([pd, vd], {}),
        ([pd, vd], dict(detector_supersampling=2, voxel_supersampling=2)),
        ((pd, vd), dict(detector_supersampling=1, voxel_supersampling=2)),
        ((pd, vd), dict(detector_supersampling=2, voxel_supersampling=1)),
    ]

    for data, kwargs in rs:
        ts.forward(*data, **kwargs)
        ts.backward(*data, **kwargs)


def test_fdk(interactive):
    if interactive:
        from tomosipo.qt import display

    pg = ts.cone(angles=100, shape=100)
    vg = ts.volume_from_projection_geometry(pg).reshape(100)
    pd = ts.data(pg)
    vd = ts.data(vg)

    # Fill the projection data with random noise:
    ts.phantom.hollow_box(vd)

    ts.forward(vd, pd)

    if interactive:
        display(vg, pg)
        display(pd)

    ts.fdk(vd, pd)

    if interactive:
        display(vd)


def test_operator():
    pg = ts.cone(angles=150, shape=100)
    vg = ts.volume(shape=100)

    A = ts.operator(vg, pg, additive=False)
    x = ts.phantom.hollow_box(ts.data(vg))

    y = A(x)
    y_ = A(np.copy(x.data))  # Test with np.array input

    assert np.sum(abs(y.data - y_.data)) == approx(0.0)

    # Test with `Data` and `np.array` again:
    x1 = A.T(y)
    x2 = A.T(y.data)

    assert np.sum(abs(x1.data - x2.data)) == approx(0.0)


def test_operator_additive():
    pg = ts.cone(angles=150, shape=100)
    vg = ts.volume(shape=100)

    A = ts.operator(vg, pg, additive=False)
    B = ts.operator(vg, pg, additive=True)

    x = ts.phantom.hollow_box(ts.data(vg))
    y = ts.data(pg)

    B(x, out=y)
    B(x, out=y)

    assert np.allclose(2 * A(x).data, y.data)
