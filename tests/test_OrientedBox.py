#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for OrientedBox."""


import unittest
import tomosipo as ts
import tomosipo.vector_calc as vc
import numpy as np
from tomosipo.Transform import random_transform
from tomosipo.OrientedBox import random_box

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
        z, y, x = (1, 0, 0), (0, 1, 0), (0, 0, 1)

        # Create unit cube:
        ob = ts.OrientedBox(1, (0, 0, 0), z, y, x)

        # Test that using a scalar position works as well.
        self.assertEqual(ob, ts.OrientedBox(1, 0, z, y, x))
        self.assertEqual(ob, ts.OrientedBox(1, 0, z, y))
        self.assertEqual(ob, ts.OrientedBox(1, 0, [z], y))
        self.assertEqual(ob, ts.OrientedBox(1, 0, z, y, x))
        self.assertEqual(ob, ts.OrientedBox((1, 1, 1), 0, z, y, x))
        self.assertEqual(ob, ts.OrientedBox((1, 1, 1), 0, z, y, x))
        self.assertEqual(ob, ts.OrientedBox((1, 1, 1), [(0, 0, 0)], z, y, x))

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

        if interactive:
            ts.display(ob1, ob2)

    def test_transform(self):
        for _ in range(10):
            box = random_box()
            T1 = random_transform()
            T2 = random_transform()

            self.assertEqual(T1(T2)(box), T1(T2(box)))
            self.assertEqual(ts.identity()(box), box)

            self.assertEqual(T1.inv(T1(box)), box)
            self.assertEqual(T1.inv(T1)(box), box)

    def test_transform_example(self):
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

        ob1 = ts.OrientedBox(1, pos, w, v, u)
        pos2 = np.stack([zero, h * np.sin(s), h * np.cos(s)], axis=1)
        ob2 = ts.OrientedBox(2, pos2, z, y, x)

        M1 = ts.from_perspective(box=ob1)
        M2 = ts.from_perspective(box=ob2)

        if interactive:
            ts.display(ob1, ob2)
            ts.display(ob1.transform(M1.matrix), ob2.transform(M1.matrix))
            ts.display(ob1.transform(M2.matrix), ob2.transform(M2.matrix))

    def test_proj_geom(self):
        # TODO: This functionality must move somewhere into the
        # codebase instead of being here in the testing ground..

        def to_boxes(pg):
            pg = pg.to_vector()

            assert pg.is_cone()

            src_pos = pg.source_positions
            det_pos = pg.detector_positions
            w = pg.detector_vs  # v points up, w points up
            u = pg.detector_us  # detector_u and u point in the same direction

            # We do not want to introduce scaling, so we normalize w and u.
            w = w / vc.norm(w)[:, None]
            u = u / vc.norm(u)[:, None]
            # This is the detector normal and has norm 1. In a lot of
            # cases, it points towards the source.
            v = vc.cross_product(u, w)

            # TODO: Warn when detector size changes during rotation.
            det_height, det_width = pg.detector_sizes[0]

            detector_box = ts.OrientedBox((det_height, 0, det_width), det_pos, w, v, u)

            # The source of course does not really have a size, but we
            # want to visualize it for now :)
            source_size = (det_width / 10,) * 3
            # We set the orientation of the source to be identical to
            # that of the detector.
            source_box = ts.OrientedBox(source_size, src_pos, w, v, u)

            return source_box, detector_box

        def vol_to_box(vg):
            return ts.OrientedBox(
                vg.size(), vg.get_center(), (1, 0, 0), (0, 1, 0), (0, 0, 1)
            )

        num_steps = 100
        s = np.linspace(0, 2 * np.pi, num_steps)
        pg = ts.cone(angles=num_steps, size=10, detector_distance=5, source_distance=5)
        vg = ts.volume()

        src_box, det_box = to_boxes(pg)
        vol_box = vol_to_box(vg)

        R = ts.rotate(det_box.pos, det_box.v, deg=10)

        if interactive:
            ts.display(src_box, vol_box, det_box)
            ts.display(src_box, (vol_box), R(det_box))

        M1 = ts.from_perspective(src_box.pos, src_box.w, src_box.v, src_box.u)

        # ts.display(src_box.transform(M1), vol_box.transform(M1), det_box.transform(M1))
