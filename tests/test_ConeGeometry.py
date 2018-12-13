#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ConeGeometry."""


import unittest
import numpy as np
import tomosipo as ts


class TestConeGeometry(unittest.TestCase):
    """Tests for ConeGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_init(self):
        """Test init."""
        interactive = False
        pg = ts.ConeGeometry()

        self.assertTrue(isinstance(pg, ts.ProjectionGeometry))

        pg = ts.ConeGeometry(angles=np.linspace(0, 1, 100))
        self.assertEqual(pg.get_num_angles(), 100)

        with self.assertRaises(ValueError):
            pg = ts.ConeGeometry(angles="asdf")

        representation = repr(pg)
        if interactive:
            print(ts.ConeGeometry())
            print(representation)

    def test_equal(self):
        """Test __eq__

        Make sure that a ConeGeometry is equal to itself.
        """

        pg = ts.ConeGeometry()
        unequal = [
            ts.ConeGeometry(angles=2),
            ts.ConeGeometry(size=5),
            ts.ConeGeometry(shape=3),
            ts.ConeGeometry(detector_distance=50),
            ts.ConeGeometry(source_distance=50),
        ]

        self.assertEqual(pg, pg)

        for u in unequal:
            self.assertNotEqual(pg, u)

    def test_cone(self):
        self.assertEqual(ts.ConeGeometry(), ts.cone())

    def test_to_vector(self):
        num_tests = 100
        for _ in range(num_tests):
            num_angles = int(np.random.uniform(1, 100))

            tupled = np.random.uniform() < 0.5
            if tupled:
                angles = np.random.uniform(0, 2 * np.pi, num_angles)
            else:
                angles = num_angles

            if tupled:
                shape = np.random.uniform(1, 100, size=2).astype(np.int)
                size = np.random.uniform(1, 100, size=2)
            else:
                shape = np.random.uniform(1, 100, size=1).astype(np.int)
                size = np.random.uniform(1, 100, size=1)

            s_dst = np.random.uniform(10, 100)
            d_dst = np.random.uniform(0, 100)

            pg1 = ts.ConeGeometry(angles, size, shape, d_dst, s_dst)
            pgv = pg1.to_vector()
            self.assertEqual(pg1.shape, pgv.shape)

    def test_to_from_astra(self):
        """Test to and from astra conversion functions

        This implementation checks the to and from astra conversion
        functions for cone beam geometries. It generates `num_tests`
        test cases and checks if the conversion to and from astra does
        not change the projection geometry.
        """
        num_tests = 100
        for _ in range(num_tests):
            num_angles = int(np.random.uniform(1, 100))

            tupled = np.random.uniform() < 0.5
            if tupled:
                angles = np.random.uniform(0, 2 * np.pi, num_angles)
            else:
                angles = num_angles

            if tupled:
                shape = np.random.uniform(1, 100, size=2).astype(np.int)
                size = np.random.uniform(1, 100, size=2)
            else:
                shape = np.random.uniform(1, 100, size=1).astype(np.int)
                size = np.random.uniform(1, 100, size=1)

            s_dst = np.random.uniform(10, 100)
            d_dst = np.random.uniform(0, 100)

            pg1 = ts.ConeGeometry(angles, size, shape, d_dst, s_dst)
            astra_pg = pg1.to_astra()
            pg2 = ts.from_astra_geometry(astra_pg)

            self.assertTrue(pg1 == pg2, msg=f"{pg1}\n\n{pg2}")

    def test_get_size(self):
        size = (1, 2)
        pg = ts.cone(size=size)
        self.assertTrue(abs(size - pg.get_size()).sum() < ts.epsilon)
