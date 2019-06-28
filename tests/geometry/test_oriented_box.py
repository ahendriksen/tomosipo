#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for OrientedBox."""

import pytest
from pytest import approx
import tomosipo as ts
import numpy as np
from tomosipo.geometry import random_transform, random_box
from tomosipo.geometry import transform


def test_init():
    """Test something."""
    z, y, x = (1, 0, 0), (0, 1, 0), (0, 0, 1)

    # Create unit cube:
    ob = ts.box(1, (0, 0, 0), z, y, x)

    # Test that using a scalar position works as well.
    assert ob == ts.box(1, 0, z, y, x)
    assert ob == ts.box(1, 0, z, y)
    assert ob == ts.box(1, 0, [z], y)
    assert ob == ts.box(1, 0, z, y, x)
    assert ob == ts.box((1, 1, 1), 0, z, y, x)
    assert ob == ts.box((1, 1, 1), 0, z, y, x)
    assert ob == ts.box((1, 1, 1), [(0, 0, 0)], z, y, x)

    N = 11
    with pytest.raises(ValueError):
        ts.box(1, [(0, 0, 0)] * 3, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N)
    with pytest.raises(ValueError):
        ts.box(1, [(0, 0, 0)] * N, [(1, 0, 0)] * 3, [(0, 1, 0)] * N, [(0, 1, 0)] * N)
    with pytest.raises(ValueError):
        ts.box(1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * 3, [(0, 1, 0)] * N)
    with pytest.raises(ValueError):
        ts.box(1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * 3)


def test_repr():
    unit_box = ts.box(size=1, pos=(0, 0, 0))
    r = """OrientedBox(
    size=(1, 1, 1),
    pos=[[0 0 0]],
    w=[[1 0 0]],
    v=[[0 1 0]],
    u=[[0 0 1]],
)"""

    assert repr(unit_box) == r


def test_eq():
    ob = ts.box(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))

    unequal = [
        ts.cone(),
        ts.box(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 99)),
        ts.box(1.1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
        ts.box(1, (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
        ts.box(1, (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)),
    ]

    assert ob == ob

    for u in unequal:
        assert ob != u


def test_size_properties():
    basis = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    ob = ts.box(1, 0, *basis)

    for s in np.random.uniform(ts.epsilon, 1, 10):
        assert approx(ob.abs_size) == ts.box(s, 0, *(basis / s)).abs_size
        assert approx(ob.abs_size) == ts.box(1 / s, 0, *(basis * s)).abs_size

    # Check that `size` fails:
    with pytest.raises(NotImplementedError), pytest.warns(DeprecationWarning):
        ob.size

    # Check that abs_size and rel_size behave under scaling:
    N = 4
    for s in np.random.normal(size=(N, 5, 3)):
        S = ts.scale(s)
        assert ob.rel_size == S(ob).rel_size
        assert ob.abs_size * abs(s) == approx(S(ob).abs_size)


def test_corners():
    # Test shape of ob.corners
    N = 11
    ob = ts.box(1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N)
    assert ob.corners.shape == (N, 8, 3)

    # Test that corners of ob2 are twice as far from the origin as
    # ob's corners.
    ob1 = ts.box(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    ob2 = ts.box(2, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    assert approx(0.0) == np.sum(abs(ob.corners * 2.0 - ob2.corners))

    ob = ts.box((1, 2, 3), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
    ob2 = ts.box((2, 4, 6), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
    # Test that corners of ob2 are twice as far from the origin as
    # ob's corners.
    assert approx(0.0) == np.sum(abs(ob.corners * 2.0 - ob2.corners))

    # Test that ob2's corners are translated by (5, 7, 11) wrt those of ob1.
    ob1 = ts.box((1, 2, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    ob2 = ts.box((1, 2, 3), (5, 7, 11), (1, 0, 0), (0, 1, 0), (0, 1, 0))
    assert approx(0.0) == np.sum(abs((ob2.corners - ob1.corners) - (5, 7, 11)))


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
    h = 3
    s = np.linspace(0, 2 * np.pi, 100)

    zero = np.zeros(s.shape)
    one = np.ones(s.shape)

    pos = np.stack([h * np.sin(s), zero, zero], axis=1)
    w = np.stack([one, zero, zero], axis=1)
    v = np.stack([zero, np.sin(s), np.cos(s)], axis=1)
    u = np.stack([zero, np.sin(s + np.pi / 2), np.cos(s + np.pi / 2)], axis=1)

    ob1 = ts.box(1, pos, w, v, u)
    pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
    ob2 = ts.box(2, pos2, w, v, u)

    if interactive:
        ts.display(ob1, ob2)


def test_transform():
    for _ in range(10):
        box = random_box()
        T1 = random_transform()
        T2 = random_transform()

        assert T1(T2)(box) == T1(T2(box))
        assert transform.identity()(box) == box

        assert T1.inv(T1(box)) == box
        assert T1.inv(T1)(box) == box


def test_display_auto_center(interactive):
    boxes = [ts.scale(.01)(random_box()) for _ in range(16)]

    if interactive:
        ts.display(*boxes)


def test_display_colors(interactive):
    boxes = [random_box() for _ in range(16)]

    if interactive:
        ts.display(*boxes)


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

    ob1 = ts.box(1, pos, w, v, u)
    pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
    ob2 = ts.box(2, pos2, z, y, x)

    M1 = ts.from_perspective(box=ob1)
    M2 = ts.from_perspective(box=ob2)

    if interactive:
        ts.display(ob1, ob2)
        ts.display(ob1.transform(M1.matrix), ob2.transform(M1.matrix))
        ts.display(ob1.transform(M2.matrix), ob2.transform(M2.matrix))
