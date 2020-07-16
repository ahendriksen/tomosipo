#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for VolumeVectorGeometry."""

import pytest
from pytest import approx
import tomosipo as ts
import numpy as np
import itertools
from tomosipo.geometry import random_transform, random_volume_vec
from tomosipo.geometry import transform
from .test_transform import vgs, translations, rotations, scalings, transforms


def test_init():
    """Test something."""
    z, y, x = (1, 0, 0), (0, 1, 0), (0, 0, 1)

    # Create unit cube:
    ob = ts.volume_vec(1, (0, 0, 0), z, y, x)

    # Test that using a scalar position works as well.
    assert ob == ts.volume_vec(1, 0, z, y, x)
    assert ob == ts.volume_vec(1, 0, z, y)
    assert ob == ts.volume_vec(1, 0, [z], y)
    assert ob == ts.volume_vec(1, 0, z, y, x)
    assert ob == ts.volume_vec((1, 1, 1), 0, z, y, x)
    assert ob == ts.volume_vec((1, 1, 1), 0, z, y, x)
    assert ob == ts.volume_vec((1, 1, 1), [(0, 0, 0)], z, y, x)

    # Check that differently shaped arguments for pos, w, v, u raise an error:
    N = 11
    with pytest.raises(ValueError):
        ts.volume_vec(
            1, [(0, 0, 0)] * 3, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N
        )
    with pytest.raises(ValueError):
        ts.volume_vec(
            1, [(0, 0, 0)] * N, [(1, 0, 0)] * 3, [(0, 1, 0)] * N, [(0, 1, 0)] * N
        )
    with pytest.raises(ValueError):
        ts.volume_vec(
            1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * 3, [(0, 1, 0)] * N
        )
    with pytest.raises(ValueError):
        ts.volume_vec(
            1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * 3
        )


def test_repr():
    unit_box = ts.volume_vec(shape=1, pos=(0, 0, 0))
    r = """VolumeVectorGeometry(
    shape=(1, 1, 1),
    pos=[[ 0.  0.  0.]],
    w=[[ 1.  0.  0.]],
    v=[[ 0.  1.  0.]],
    u=[[ 0.  0.  1.]],
)"""

    assert repr(unit_box) == r


def test_eq():
    ob = ts.volume_vec(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))

    unequal = [
        ts.cone(),
        ts.volume_vec(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 99)),
        ts.volume_vec(2, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
        ts.volume_vec(1, (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
        ts.volume_vec(1, (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)),
    ]

    assert ob == ob

    for u in unequal:
        assert ob != u


@pytest.mark.parametrize(
    "vg, T, S, R", itertools.product(vgs, translations, scalings, rotations)
)
def test_properties_under_transformations(vg, T, S, R):
    TSR = T * S * R
    P = ts.from_perspective(box=vg)
    assert TSR * vg == TSR * vg
    assert (TSR * vg)[0, 0, 0, 0] == TSR * vg[0, 0, 0, 0]
    assert (TSR * vg)[-1, -1, -1, -1] == TSR * vg[-1, -1, -1, -1]
    assert (S * R * P * vg).pos == approx((P * vg).pos)
    assert (T * vg).w == approx(vg.w)
    assert (T * vg).v == approx(vg.v)
    assert (T * vg).u == approx(vg.u)
    assert (TSR * vg).shape == vg.shape
    assert (T * vg).size == vg.size
    assert (T * vg).sizes == approx(vg.sizes)
    assert (T * vg).voxel_sizes == approx(vg.voxel_sizes)
    assert (T * vg).voxel_size == vg.voxel_size
    # corners?
    # lowerleftcorner?


def test_lower_left_corner():
    assert approx(ts.volume_vec(1, pos=0).lower_left_corner) == [(-0.5, -0.5, -0.5)]
    assert approx(ts.volume_vec(1, pos=(0.5, 0.5, 0.5)).lower_left_corner) == [
        (0, 0, 0)
    ]


def test_get_item():
    vg = random_volume_vec()
    assert vg[0] == vg
    assert vg[:1] == vg
    assert vg[-1] == vg
    assert vg[:] == vg
    assert ts.concatenate([vg, vg])[0] == vg
    assert ts.concatenate([vg, vg])[1] == vg

    assert ts.volume_vec(shape=3)[:, 1, 1, 1] == ts.volume_vec(shape=1)
    T = random_transform()
    assert T * vg[:, 1, 2, 3] == (T * vg)[:, 1, 2, 3]
    assert T * vg[:, :1, :2, :3] == (T * vg)[:, :1, :2, :3]
    assert T * vg[:, 1:, 2:, 3:] == (T * vg)[:, 1:, 2:, 3:]
    with pytest.raises(ValueError):
        vg[1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        vg[...]


def test_size():
    vg = ts.geometry.random_volume().to_vec()
    # Non-uniform scaling
    S = ts.scale(abs(np.random.normal(size=(3, 3))))
    with pytest.raises(ValueError):
        (S * vg).size
    with pytest.raises(ValueError):
        (S * vg).voxel_size


def test_corners():
    # Test shape of ob.corners
    N = 11
    vg = ts.volume_vec(
        1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N
    )
    assert vg.corners.shape == (N, 8, 3)

    # Test that corners of vg2 are twice as far from the origin as
    # vg's corners.
    vg1 = ts.volume_vec(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    vg2 = ts.volume_vec(2, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    assert approx(0.0) == np.sum(abs(vg.corners * 2.0 - vg2.corners))

    vg = ts.volume_vec((1, 2, 3), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
    vg2 = ts.volume_vec((2, 4, 6), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
    # Test that corners of vg2 are twice as far from the origin as
    # vg's corners.
    assert approx(0.0) == np.sum(abs(vg.corners * 2.0 - vg2.corners))

    # Test that vg2's corners are translated by (5, 7, 11) wrt those of vg1.
    vg1 = ts.volume_vec((1, 2, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    vg2 = ts.volume_vec((1, 2, 3), (5, 7, 11), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    assert approx(0.0) == np.sum(abs((vg2.corners - vg1.corners) - (5, 7, 11)))


def test_display(interactive):
    """Test display function

    We display two boxes:

    - one box jumps up and down in a sinusoidal movement while it
      is rotating anti-clockwise.
    - another box is twice as large and rotates around origin in
      the axial plane while rotating itself as well.

    :returns:
    :rtype:

    """
    # TODO: Implement display for volume_vec geometry!
    h = 3
    s = np.linspace(0, 2 * np.pi, 100)

    zero = np.zeros(s.shape)
    one = np.ones(s.shape)

    pos = np.stack([h * np.sin(s), zero, zero], axis=1)
    w = np.stack([one, zero, zero], axis=1)
    v = np.stack([zero, np.sin(s), np.cos(s)], axis=1)
    u = np.stack([zero, np.sin(s + np.pi / 2), np.cos(s + np.pi / 2)], axis=1)

    ob1 = ts.volume_vec(1, pos, w, v, u)
    pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
    ob2 = ts.volume_vec(2, pos2, w, v, u)

    if interactive:
        from tomosipo.qt import display

        display(ob1, ob2)


def test_transform():
    for _ in range(10):
        vg = random_volume_vec()
        T1 = random_transform()
        T2 = random_transform()

        assert (T1 * T2) * vg == T1 * (T2 * vg)
        assert transform.identity() * vg == vg

        assert T1.inv * (T1 * vg) == vg
        assert (T1.inv * T1) * vg == vg


def test_display_auto_center(interactive):
    vgs = [ts.scale(0.01) * random_volume_vec() for _ in range(16)]

    if interactive:
        from tomosipo.qt import display

        display(*vgs)


def test_display_colors(interactive):
    vgs = [random_volume_vec() for _ in range(16)]

    if interactive:
        from tomosipo.qt import display

        display(*vgs)


def test_transform_example(interactive):
    h = 3
    s = np.linspace(0, 2 * np.pi, 100)

    zero = np.zeros(s.shape)
    one = np.ones(s.shape)

    pos = np.stack([h * np.sin(s), zero, zero], axis=1)
    z = np.stack([one, zero, zero], axis=1)
    y = np.stack([zero, one, zero], axis=1)
    x = np.stack([zero, zero, one], axis=1)

    w = np.stack([one, zero, zero], axis=1)
    v = np.stack([zero, np.sin(s), np.cos(s)], axis=1)
    u = np.stack([zero, np.sin(s + np.pi / 2), np.cos(s + np.pi / 2)], axis=1)

    vg1 = ts.volume_vec(1, pos, w, v, u)
    pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
    vg2 = ts.volume_vec(2, pos2, z, y, x)

    M1 = ts.from_perspective(box=vg1)
    M2 = ts.from_perspective(box=vg2)

    if interactive:
        from tomosipo.qt import display

        display(vg1, vg2)
        display(M1 * vg1, M1 * vg2)
        display(M2 * vg1, M2 * vg2)
