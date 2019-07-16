#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Transform."""


import tomosipo as ts
from tomosipo.geometry import random_box
import numpy as np
from tomosipo.geometry import transform


def test_identity():
    T = transform.identity()
    box = ts.box(1, (-1, -2, -3), (2, 3, 5), (7, 11, 13), (17, 19, 23))
    assert T * box == box
    for _ in range(5):
        params = np.random.normal(0, 1, size=(5, 3))
        box = ts.box(*params)
        assert T * box == box


def test_eq():
    T = transform.identity()
    assert T == T
    for _ in range(5):
        params = np.random.normal(0, 1, size=(1, 3))
        T = ts.translate(*params)
        assert T == T


def test_translate():
    """Test something."""
    N = 10

    # "Simple case": we check that the position of a rotated box
    # changes correctly.
    R = ts.rotate((2, 3, 5), axis=(7, 11, -13), deg=15)
    original = R * ts.box(size=(3, 4, 5), pos=(2, 3, 5))

    t = np.array((3, 4, 5))
    T = ts.translate(t)
    print(T * original)
    print(ts.box((3, 4, 5), t + (2, 3, 5), original.w, original.v, original.u))
    assert T * original == ts.box(
        (3, 4, 5), t + (2, 3, 5), original.w, original.v, original.u
    )

    # General case: check that translation by t1 and t2 is the
    # same as translation by t1 + t2.
    for t1, t2 in np.random.normal(size=(N, 2, 3)):
        box = random_box()
        T1 = ts.translate(t1)
        T2 = ts.translate(t2)
        T = ts.translate(t1 + t2)

        assert (T1 * T2) * box == T1 * (T2 * box)
        assert T1 * T2 == T


def test_scale():
    unit = ts.box(size=1, pos=0)
    assert ts.scale(5) * unit == ts.box(5, 0)
    assert ts.scale((5, 3, 2)) * unit == ts.box(size=(5, 3, 2), pos=0)

    # Check that scaling by s1 and s2 is the same as scaling by s1 + s2.
    N = 10
    for s1, s2 in np.random.normal(size=(N, 2, 3)):
        box = random_box()
        S1 = ts.scale(s1)
        S2 = ts.scale(s2)
        S = ts.scale(s1 * s2)

        assert (S1 * S2) * box == S1 * (S2 * box)
        assert S1 * S2 == S


def test_rotate(interactive):
    N = 50
    for p, axis in np.random.normal(size=(N, 2, 3)):
        angle = 2 * np.pi * np.random.normal()
        # Test handedness by inverting the angle and also by inverting the rotation axis.
        assert ts.rotate(p, axis, rad=angle, right_handed=True) == ts.rotate(
            p, axis, rad=-angle, right_handed=False
        )
        assert ts.rotate(p, axis, rad=angle, right_handed=True) == ts.rotate(
            p, -axis, rad=angle, right_handed=False
        )
        # Ensure that adding 2*pi to the angle does not affect the rotation
        assert ts.rotate(p, axis, rad=angle, right_handed=True) == ts.rotate(
            p, axis, rad=angle + 2 * np.pi, right_handed=True
        )
        # Ensure that scaling the rotation axis does not affect rotation
        assert ts.rotate(p, axis, rad=angle, right_handed=True) == ts.rotate(
            p, 2 * axis, rad=angle, right_handed=True
        )

    # Check that rotating by theta1 and theta2 is the same as
    # rotating by theta1 + theta2.
    N = 10
    for theta1, theta2 in np.random.normal(size=(N, 2)):
        box = random_box()
        axis = np.random.normal(size=3)
        pos = np.random.normal(size=3)
        R1 = ts.rotate(pos, axis, rad=theta1)
        R2 = ts.rotate(pos, axis, rad=theta2)
        R = ts.rotate(pos, axis, rad=theta1 + theta2)

        assert (R1 * R2) * box == R1 * (R2 * box)
        assert R1 * R2 == R

    # Show a box rotating around the Z-axis:
    box = ts.box((5, 2, 2), 0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
    # top_box is located above (Z-axis) the box to show in
    # which direction the Z-axis points
    top_box = ts.box(.5, (3, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))

    s = np.linspace(0, 360, 361, endpoint=True)
    R = ts.rotate((0, 0, 0), (1, 0, 0), deg=s, right_handed=True)

    if interactive:
        ts.display(R * box, top_box)


def test_perspective():
    """Test to_perspective and from_perspective functions.

    """

    # Unit cube
    unit = ts.box(1, 0)

    N = 50
    for t, p, axis in np.random.normal(size=(N, 3, 3)):
        angle = 2 * np.pi * np.random.normal()
        T = ts.translate(t)
        R = ts.rotate(p, axis, rad=angle)
        # We now have a unit cube on some random location:
        random_box = (R * T) * unit

        # Check that we can move the unit cube to the random box:
        to_random_box = ts.to_perspective(box=random_box)
        assert random_box == to_random_box * unit

        # Check that we can move the random box to the unit cube:
        to_unit_cube = ts.from_perspective(box=random_box)
        assert unit == to_unit_cube * random_box

        # Check that to_unit_cube is the inverse of to_random_box
        assert to_random_box * to_unit_cube == transform.identity()

        # Check that we can use pos, w, v, u parameters:
        to_random_box2 = ts.to_perspective(
            random_box.pos, random_box.w, random_box.v, random_box.u
        )
        assert to_random_box == to_random_box2
