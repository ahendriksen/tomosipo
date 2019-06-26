#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Transform."""


import unittest
import tomosipo as ts
from tomosipo.geometry import random_box
import numpy as np

interactive = False


class TestTransform(unittest.TestCase):
    """Tests for Transform."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_identity(self):
        T = ts.identity()
        box = ts.box(1, (-1, -2, -3), (2, 3, 5), (7, 11, 13), (17, 19, 23))
        self.assertEqual(T(box), box)
        for _ in range(5):
            params = np.random.normal(0, 1, size=(5, 3))
            box = ts.box(*params)
            self.assertEqual(T(box), box)

    def test_eq(self):
        T = ts.identity()
        self.assertEqual(T, T)
        for _ in range(5):
            params = np.random.normal(0, 1, size=(1, 3))
            T = ts.translate(*params)
            self.assertEqual(T, T)

    def test_translate(self):
        """Test something."""
        N = 10

        # "Simple case": we check that the position of a rotated box
        # changes correctly.
        rotate = ts.rotate((2, 3, 5), axis=(7, 11, -13), deg=15)
        original = rotate(ts.box(size=(3, 4, 5), pos=(2, 3, 5)))

        t = np.array((3, 4, 5))
        T = ts.translate(t)
        self.assertEqual(
            T(original),
            ts.box((3, 4, 5), t + (2, 3, 5), original.w, original.v, original.u),
        )

        # General case: check that translation by t1 and t2 is the
        # same as translation by t1 + t2.
        for t1, t2 in np.random.normal(size=(N, 2, 3)):
            box = random_box()
            T1 = ts.translate(t1)
            T2 = ts.translate(t2)
            T = ts.translate(t1 + t2)

            self.assertEqual(T1(T2)(box), T1(T2(box)))
            self.assertEqual(T1(T2), T)

    def test_scale(self):
        unit = ts.box(size=1, pos=0)
        self.assertEqual(ts.scale(5)(unit), ts.box(5, 0))
        self.assertEqual(ts.scale((5, 3, 2))(unit), ts.box(size=(5, 3, 2), pos=0))

        # Check that scaling by s1 and s2 is the same as scaling by s1 + s2.
        N = 10
        for s1, s2 in np.random.normal(size=(N, 2, 3)):
            box = random_box()
            S1 = ts.scale(s1)
            S2 = ts.scale(s2)
            S = ts.scale(s1 * s2)

            self.assertEqual(S1(S2)(box), S1(S2(box)))
            self.assertEqual(S1(S2), S)

    def test_rotate(self):
        # TODO: Test continuity as axis approaches (0, 0, 1)
        N = 50
        for p, axis in np.random.normal(size=(N, 2, 3)):
            angle = 2 * np.pi * np.random.normal()
            # Test handedness by inverting the angle and also by inverting the rotation axis.
            self.assertEqual(
                ts.rotate(p, axis, rad=angle, right_handed=True),
                ts.rotate(p, axis, rad=-angle, right_handed=False),
            )
            self.assertEqual(
                ts.rotate(p, axis, rad=angle, right_handed=True),
                ts.rotate(p, -axis, rad=angle, right_handed=False),
            )
            # Ensure that adding 2*pi to the angle does not affect the rotation
            self.assertEqual(
                ts.rotate(p, axis, rad=angle, right_handed=True),
                ts.rotate(p, axis, rad=angle + 2 * np.pi, right_handed=True),
            )
            # Ensure that scaling the rotation axis does not affect rotation
            self.assertEqual(
                ts.rotate(p, axis, rad=angle, right_handed=True),
                ts.rotate(p, 2 * axis, rad=angle, right_handed=True),
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

            self.assertEqual(R1(R2)(box), R1(R2(box)))
            self.assertEqual(R1(R2), R)

        # Show a box rotating around the Z-axis:
        box = ts.box((5, 2, 2), 0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
        # top_box is located above (Z-axis) the box to show in
        # which direction the Z-axis points
        top_box = ts.box(.5, (3, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))

        s = np.linspace(0, 360, 361, endpoint=True)
        R = ts.rotate((0, 0, 0), (1, 0, 0), deg=s, right_handed=True)

        if interactive:
            ts.display(R(box), top_box)

    def test_perspective(self):
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
            random_box = R(T)(unit)

            # Check that we can move the unit cube to the random box:
            to_random_box = ts.to_perspective(box=random_box)
            self.assertEqual(random_box, to_random_box(unit))

            # Check that we can move the random box to the unit cube:
            to_unit_cube = ts.from_perspective(box=random_box)
            self.assertEqual(unit, to_unit_cube(random_box))

            # Check that to_unit_cube is the inverse of to_random_box
            self.assertEqual(to_random_box(to_unit_cube), ts.identity())

            # Check that we can use pos, w, v, u parameters:
            to_random_box2 = ts.to_perspective(
                random_box.pos, random_box.w, random_box.v, random_box.u
            )
            self.assertEqual(to_random_box, to_random_box2)
