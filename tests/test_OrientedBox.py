#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for OrientedBox."""


import unittest
import tomosipo as ts
import numpy as np

interactive = False


class TestOrientedBox(unittest.TestCase):
    """Tests for OrientedBox."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_init(self):
        """Test something."""

        ob = ts.OrientedBox(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
        print(ob)

        N = 11
        with self.assertRaises(ValueError):
            ts.OrientedBox(
                1, [(0, 0, 0)] * 3, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N
            )
        with self.assertRaises(ValueError):
            ts.OrientedBox(
                1, [(0, 0, 0)] * N, [(1, 0, 0)] * 3, [(0, 1, 0)] * N, [(0, 1, 0)] * N
            )
        with self.assertRaises(ValueError):
            ts.OrientedBox(
                1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * 3, [(0, 1, 0)] * N
            )
        with self.assertRaises(ValueError):
            ts.OrientedBox(
                1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * 3
            )

    def test_eq(self):
        ob = ts.OrientedBox(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))

        unequal = [
            ts.cone(),
            ts.OrientedBox(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 99)),
            ts.OrientedBox(1.1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
            ts.OrientedBox(1, (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
            ts.OrientedBox(1, (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0)),
        ]

        self.assertEqual(ob, ob)

        for u in unequal:
            self.assertNotEqual(ob, u)

    def test_corners(self):
        # Test shape of ob.corners
        N = 11
        ob = ts.OrientedBox(
            1, [(0, 0, 0)] * N, [(1, 0, 0)] * N, [(0, 1, 0)] * N, [(0, 1, 0)] * N
        )
        self.assertEqual(ob.corners.shape, (N, 8, 3))

        # Test that corners of ob2 are twice as far from the origin as
        # ob's corners.
        ob1 = ts.OrientedBox(1, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
        ob2 = ts.OrientedBox(2, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
        self.assertAlmostEqual(0.0, np.sum(abs(ob.corners * 2.0 - ob2.corners)))

        ob = ts.OrientedBox((1, 2, 3), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
        ob2 = ts.OrientedBox((2, 4, 6), (0, 0, 0), (2, 3, 5), (7, 11, 13), (17, 19, 23))
        # Test that corners of ob2 are twice as far from the origin as
        # ob's corners.
        self.assertAlmostEqual(0.0, np.sum(abs(ob.corners * 2.0 - ob2.corners)))

        # Test that ob2's corners are translated by (5, 7, 11) wrt those of ob1.
        ob1 = ts.OrientedBox((1, 2, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))
        ob2 = ts.OrientedBox((1, 2, 3), (5, 7, 11), (1, 0, 0), (0, 1, 0), (0, 1, 0))
        self.assertAlmostEqual(
            0.0, np.sum(abs((ob2.corners - ob1.corners) - (5, 7, 11)))
        )

    def test_display(self):
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

        ob1 = ts.OrientedBox(1, pos, w, v, u)
        pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
        ob2 = ts.OrientedBox(2, pos2, w, v, u)

        ts.display(ob1, ob2)
