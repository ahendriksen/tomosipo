#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for operator."""

import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
import itertools
from .geometry.test_transform import scalings, rotations, translations


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
        ts.forward(*data, **kwargs)
        ts.backward(*data, **kwargs)


def test_fdk(interactive):
    if interactive:
        from tomosipo.qt import display

    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=100, shape=100)
    vg = ts.volume(shape=100)
    pd = ts.data(pg)
    vd = ts.data(vg)

    ts.phantom.hollow_box(vd)
    ts.forward(vd, pd)

    if interactive:
        display(vg, pg)
        display(pd)

    ts.fdk(vd, pd)

    if interactive:
        display(vd)


def test_operator():
    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=150, shape=100)
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
    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=150, shape=100)
    vg = ts.volume(shape=100)

    A = ts.operator(vg, pg, additive=False)
    B = ts.operator(vg, pg, additive=True)

    x = ts.phantom.hollow_box(ts.data(vg))
    y = ts.data(pg)

    B(x, out=y)
    B(x, out=y)

    assert np.allclose(2 * A(x).data, y.data)


@pytest.mark.parametrize(
    "S, T, R", itertools.product(scalings, translations, rotations)
)
def test_operator_volume_vector(S, T, R):
    """Tests ts.operator with volume vector geometries

    This test ensures that forward and backprojection using volume
    vector geometries works as expected.

    Specifically, for any geometric transformation T that is a
    combination of rotation and translation, we must have:

    ts.operator(T * vg, pg) == ts.operator(vg, T^-1 * pg),

    that is: the two operators compute the same function.

    """

    pg = S * ts.parallel(angles=64, shape=96, size=1.5).to_vec()
    vg = S * ts.volume(shape=64, size=1)

    x = np.random.normal(size=(64, 64, 64)).astype(np.float32)

    # Generate operator by rotating the volume:
    A_v = ts.operator(T * R * vg.to_vec(), pg)
    # And by "unrotating" the projection geometry
    A_p = ts.operator(vg, (T * R).inv * pg)

    # We should get the same forward and backprojection
    assert np.allclose(A_v(x), A_p(x))
    assert np.allclose(A_v.T(A_v(x)), A_p.T(A_p(x)))
