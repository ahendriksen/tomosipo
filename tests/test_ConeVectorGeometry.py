#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ConeVectorGeometry."""


import unittest
import astra
import numpy as np
import tomosipo as ts
from tomosipo.Transform import random_transform
from tomosipo.ConeVectorGeometry import random_cone_vec


class TestConeVectorGeometry(unittest.TestCase):
    """Tests for ConeVectorGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_init(self):
        """Test something."""
        interactive = False
        vectors = np.arange(30).reshape((10, 3))

        # This should not raise an exception
        pg = ts.cone_vec(1, vectors, vectors, vectors, vectors)

        representation = repr(pg)
        if interactive:
            print(representation)

        with self.assertRaises(ValueError):
            ts.cone_vec(0, vectors, vectors, vectors, vectors)
        with self.assertRaises(ValueError):
            vecs = np.arange(20).reshape((10, 2))
            ts.cone_vec(1, vecs, vecs, vecs, vecs)
        with self.assertRaises(ValueError):
            ts.cone_vec(1, vectors, vectors, vectors, vecs)

    def test_equal(self):
        """Test __eq__

        Make sure that a ConeVectorGeometry is equal to itself.
        """

        pos = np.array([(0, 0, 0)])
        pg = ts.cone_vec(10, pos, pos, pos, pos)
        unequal = [
            ts.cone_vec(9, pos, pos, pos, pos),
            ts.cone_vec(10, pos + 1, pos, pos, pos),
            ts.cone_vec(10, pos, pos + 1, pos, pos),
            ts.cone_vec(10, pos, pos, pos + 1, pos),
            ts.cone_vec(10, pos, pos, pos, pos + 1),
            ts.cone(angles=2),
            ts.cone(),
            ts.volume(),
        ]

        self.assertEqual(pg, pg)

        for u in unequal:
            self.assertNotEqual(pg, u)

    def test_to_from_astra(self):
        nrows = 30
        num_tests = 100
        for _ in range(num_tests):
            shape = tuple(int(np.random.uniform(1, 100)) for _ in range(2))
            v1, v2, v3, v4 = (np.random.uniform(size=(nrows, 3)) for _ in range(4))

            pg = ts.cone_vec(shape, v1, v2, v3, v4)
            astra_pg = pg.to_astra()
            self.assertEqual(pg, ts.from_astra_geometry(astra_pg))

    def test_project_point_detector_spacing(self):
        """Test project_point with detector spacing

        Test whether projection_point works with non-uniform detector spacing.
        """
        shape = (10, 40)
        size = (30, 80)

        pg = ts.cone(1, size, shape, source_distance=10).to_vector()
        self.assertAlmostEqual(np.abs(pg.project_point((0, 0, 0))).sum(), 0)

        self.assertAlmostEqual(
            np.abs(pg.project_point((3.0, 0, 0)) - [1.0, 0.0]).sum(), 0
        )
        self.assertAlmostEqual(
            np.abs(pg.project_point((0, 0, 2.0)) - [0.0, 1.0]).sum(), 0
        )

        shape = (100, 400)
        pg = ts.cone(1, size, shape, source_distance=10).to_vector()
        self.assertAlmostEqual(np.abs(pg.project_point((0, 0, 0))).sum(), 0)
        self.assertAlmostEqual(
            np.abs(pg.project_point((0.3, 0, 0)) - [1.0, 0.0]).sum(), 0
        )
        self.assertAlmostEqual(
            np.abs(pg.project_point((0, 0, 0.2)) - [0.0, 1.0]).sum(), 0
        )

    def test_detector_sizes(self):
        size = (1, 1)
        pg = ts.cone(angles=5, size=size).to_vector()
        self.assertTrue(abs(size - pg.detector_sizes).sum() < ts.epsilon)
        for _ in range(10):
            new_shape = np.random.uniform(1, 100, size=2).astype(np.int)
            # Change the number of detector pixels and test if the
            # change in size is proportional
            pg2 = pg.reshape(new_shape)
            self.assertTrue(
                abs(size * new_shape - pg2.detector_sizes).sum() < ts.epsilon
            )

    def test_get_corners(self):
        # TODO: This test deserves better..
        size = (1, 1)
        pg1 = ts.cone(angles=5, size=size).to_vector()

        self.assertEqual(pg1.get_corners().shape, (4, 5, 3))

    def test_get_source_positions(self):
        pg1 = ts.cone(source_distance=1).to_vector()
        pg2 = ts.cone(source_distance=2).to_vector()

        source_diff = pg1.get_source_positions() * 2 - pg2.get_source_positions()
        self.assertTrue(np.all(abs(source_diff) < ts.epsilon))

    def test_transform(self):

        for _ in range(10):
            pg = random_cone_vec()
            T1 = random_transform()
            T2 = random_transform()

            self.assertEqual(T1(T2)(pg), T1(T2(pg)))
            self.assertEqual(ts.identity()(pg), pg)
