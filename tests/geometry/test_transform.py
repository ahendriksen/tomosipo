#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Transform."""

import pytest
import tomosipo as ts
from tomosipo.geometry import random_volume_vec
import numpy as np
from tomosipo.geometry import transform
import itertools


vgs = [
    ts.volume_vec(shape=1, pos=0),
    ts.volume_vec(shape=(3, 5, 7), pos=0),
    ts.volume_vec(shape=(3, 5, 7), pos=(0.1, -0.3, 4.5)),
    random_volume_vec(),
]

translations = [
    ts.translate((0, 0, 0)),  # identity
    ts.translate(*np.random.normal(0, 1, size=(1, 3))),
]

rotations = [
    ts.rotate(pos=0, axis=(1, 0, 0), rad=0),  # identity
    ts.rotate(pos=0, axis=(1, 0, 0), rad=1.0),
    ts.rotate(pos=0, axis=np.random.normal(size=3), deg=np.random.normal()),
    ts.rotate(
        pos=0, axis=np.random.normal(size=3), deg=np.random.normal(), right_handed=False
    ),
]

scalings = [
    ts.scale(1),  # identity
    ts.scale(abs(np.random.normal())),  # same random scaling in each direction
    ts.scale(
        abs(np.random.normal(size=3))
    ),  # different random scaling in each direction
]

transforms = translations + rotations + scalings


@pytest.mark.parametrize(
    "vg, T, R, S", itertools.product(vgs, translations, rotations, scalings)
)
def test_equations(vg, T, R, S):
    # identity:
    id = transform.identity()
    assert id * vg == vg
    assert id * S == S
    assert id * T == T
    assert id * R == R
    assert id * (S * T * R) == S * T * R
    # Associativity of group action:
    assert S * (T * (R * vg)) == ((S * T) * R) * vg
    # Associativity:
    assert S * (T * R) == (S * T) * R
    # inverse
    for t in [id, S, T, R, S * T * R]:
        assert id == t * t.inv
        assert id == t.inv * t
    assert (T * S * R).inv == R.inv * S.inv * T.inv


@pytest.mark.parametrize("vg", vgs)
def test_identity(vg):
    T = transform.identity()
    assert T * vg == vg


@pytest.mark.parametrize("T", transforms)
def test_eq(T):
    assert T == T
    assert T != ts.translate((0.1, 0.1, 0.1)) * T
    assert T != ts.scale(0.1) * T
    assert T != ts.rotate(pos=0, axis=(1, 0, 0), deg=1) * T


@pytest.mark.parametrize("T", translations)
def test_get_item(T):
    assert T[0] == T
    assert T[:1] == T
    assert ts.concatenate([T, T])[0] == T
    assert ts.concatenate([T, T])[1] == T
    with pytest.raises(TypeError):
        T[0, 0]
        T[...]


def test_translate_simple_case():
    # "Simple case": we check that the position of a rotated box
    # changes correctly.
    R = ts.rotate((2, 3, 5), axis=(7, 11, -13), deg=15)
    original = R * ts.volume_vec(shape=(3, 4, 5), pos=(2, 3, 5))

    t = np.array((3, 4, 5))
    T = ts.translate(t)

    assert T * original == ts.volume_vec(
        shape=(3, 4, 5), pos=t + (2, 3, 5), w=original.w, v=original.v, u=original.u
    )


def test_translate():
    # General case: check that translation by t1 and t2 is the
    # same as translation by t1 + t2.
    N = 5
    for t1, t2 in np.random.normal(size=(N, 2, 3)):
        vg = random_volume_vec()
        T1 = ts.translate(t1)
        T2 = ts.translate(t2)
        T = ts.translate(t1 + t2)

        assert (T1 * T2) * vg == T1 * (T2 * vg)
        assert T1 * T2 == T
        assert T1 * T2 == T2 * T1


def test_scale_simple_case():
    unit = ts.volume_vec(shape=1, pos=0)
    # Uniform scaling
    s = abs(np.random.normal())
    scaled_unit = ts.volume_vec(
        shape=1, pos=0, w=s * unit.w, v=s * unit.v, u=s * unit.u
    )
    assert ts.scale(s) * unit == scaled_unit
    # Non-uniform scaling
    s = abs(np.random.normal(size=3))
    scaled_unit = ts.volume_vec(
        shape=1, pos=0, w=s[0] * unit.w, v=s[1] * unit.v, u=s[2] * unit.u
    )
    assert ts.scale(s) * unit == scaled_unit


def test_scale():
    # Check that scaling by s1 and s2 is the same as scaling by s1 * s2.
    N = 10
    for s1, s2 in abs(np.random.normal(size=(N, 2, 3))):
        vg = random_volume_vec()
        S1 = ts.scale(s1)
        S2 = ts.scale(s2)
        S = ts.scale(s1 * s2)

        assert (S1 * S2) * vg == S1 * (S2 * vg)
        assert S1 * S2 == S


def test_rotate_inversion_of_angle_axis_handedness():
    N = 10
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


def test_rotate_adding_angles():
    # Check that rotating by theta1 and theta2 is the same as
    # rotating by theta1 + theta2.
    N = 10
    for theta1, theta2 in np.random.normal(size=(N, 2)):
        vg = random_volume_vec()
        axis = np.random.normal(size=3)
        pos = np.random.normal(size=3)
        R1 = ts.rotate(pos, axis, rad=theta1)
        R2 = ts.rotate(pos, axis, rad=theta2)
        R = ts.rotate(pos, axis, rad=theta1 + theta2)

        assert (R1 * R2) * vg == R1 * (R2 * vg)
        assert R1 * R2 == R


def test_rotate_visually(interactive):
    # Show a box rotating around the Z-axis:
    vg = ts.volume_vec(shape=(5, 2, 2), pos=0)
    # top_box is located above (Z-axis) the box to show in
    # which direction the Z-axis points
    top_vg = ts.volume_vec(shape=1, pos=(3, 0, 0))

    s = np.linspace(0, 360, 361, endpoint=True)
    R = ts.rotate((0, 0, 0), (1, 0, 0), deg=s, right_handed=True)

    if interactive:
        from tomosipo.qt import display

        display(R * vg, top_vg)


def test_perspective():
    """Test to_perspective and from_perspective functions.

    """
    # Unit cube
    unit = ts.volume_vec(shape=1, pos=0)

    N = 10
    for t, p, axis in np.random.normal(size=(N, 3, 3)):
        angle = 2 * np.pi * np.random.normal()
        T = ts.translate(t)
        R = ts.rotate(p, axis, rad=angle)
        # We now have a unit cube on some random location:
        random_vg = (R * T) * unit

        # Check that we can move the unit cube to the random box:
        to_random_vg = ts.to_perspective(box=random_vg)
        assert random_vg == to_random_vg * unit

        # Check that we can move the random box to the unit cube:
        to_unit_cube = ts.from_perspective(box=random_vg)
        assert unit == to_unit_cube * random_vg

        # Check that to_unit_cube is the inverse of to_random_vg
        assert to_random_vg * to_unit_cube == transform.identity()

        # Check that we can use pos, w, v, u parameters:
        to_random_vg2 = ts.to_perspective(
            random_vg.pos, random_vg.w, random_vg.v, random_vg.u
        )
        assert to_random_vg == to_random_vg2
