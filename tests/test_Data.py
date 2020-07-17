#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Data."""


import unittest
import numpy as np
import tomosipo as ts


class TestData(unittest.TestCase):
    """Tests for Data."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_init(self):
        """Test data creation."""
        pg = ts.cone(angles=10, shape=20, cone_angle=1 / 2)
        vg = ts.volume().reshape(10)
        pd = ts.data(pg, 0)

        proj_shape = pd.data.shape

        # Should warn when data is silently converted to float32.
        with self.assertWarns(UserWarning):
            ts.data(pg, np.ones(proj_shape, dtype=np.float64))
        with self.assertWarns(UserWarning):
            ts.data(vg, np.ones(vg.shape, dtype=np.float64))

        # Should warn when data is made contiguous.
        with self.assertWarns(UserWarning):
            p_data = np.ones(
                (pg.det_shape[0] * 2, pg.num_angles, pg.det_shape[1]), dtype=np.float32
            )
            ts.data(pg, p_data[::2, ...])
        with self.assertWarns(UserWarning):
            v_data = np.ones((vg.shape[0] * 2, *vg.shape[1:]), dtype=np.float32)
            ts.data(vg, v_data[::2, ...])

        # Should raise an error when geometry is not convertible to ASTRA:
        with self.assertRaises(TypeError):
            ts.data(vg.to_vec())

        ts.data(pg, np.ones(proj_shape, dtype=np.float32))
        ts.data(vg, np.ones(vg.shape, dtype=np.float32))

    def test_with(self):
        """Test that data in a with statement gets cleaned up

        Also, that using the with statement works.
        """
        # TODO: Remove this, segfaults are no longer a feature...
        seg_fault = True
        pg = ts.cone(cone_angle=1 / 2)
        vg = ts.volume()

        with ts.data(pg, 0) as pd, ts.data(vg, 0) as vd:
            proj = pd.data
            vol = vd.data

        # No test to check that the code below segfaults, but it does :)
        # You can run the code..
        if seg_fault:
            proj[:] = 0
            vol[:] = 0

    def test_data(self):
        """Test data.data property
        """

        pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2).reshape(10)
        d = ts.data(pg, 0)

        self.assertTrue(np.all(abs(d.data) < ts.epsilon))
        d.data[:] = 1.0
        self.assertTrue(np.all(abs(d.data - 1) < ts.epsilon))

    def test_is_volume_projection(self):
        self.assertTrue(
            ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2)).is_projection()
        )
        self.assertFalse(
            ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2)).is_volume()
        )
        self.assertTrue(ts.data(ts.volume()).is_volume())
        self.assertFalse(ts.data(ts.volume()).is_projection())

    def test_init_idempotency(self):
        """Test that ts.data can be used idempotently

        In numpy, you do `np.array(np.array([1, 1]))' to cast a list
        to a numpy array. The second invocation of `np.array'
        basically short-circuits and returns the argument.

        Here, we test that we have the same behaviour for `ts.data'.
        """
        vg = ts.volume(shape=10)

        vd = ts.data(vg)
        vd_ = ts.data(vg, vd)

        self.assertEqual(id(vd), id(vd_))

        with self.assertRaises(ValueError):
            ts.data(ts.volume(), vd)

        pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)
        # TODO: Implement and use .copy()
        pg_ = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)

        pd = ts.data(pg)
        pd_ = ts.data(pg_, pd)

        self.assertEqual(id(pd), id(pd_))

        with self.assertRaises(ValueError):
            ts.data(vg, pd)

        with self.assertRaises(ValueError):
            ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2), pd)
