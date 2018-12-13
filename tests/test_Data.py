#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Data."""


import unittest
import numpy as np
import astra
import warnings
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
        pg = ts.cone(angles=10, shape=20)
        vg = ts.VolumeGeometry().reshape(10)
        pd = ts.Data(pg, 0)

        proj_shape = pd.get().shape

        # Should warn when data is silently converted to float32.
        with self.assertWarns(UserWarning):
            ts.Data(pg, np.ones(proj_shape, dtype=np.float64))
        with self.assertWarns(UserWarning):
            ts.Data(vg, np.ones(vg.shape, dtype=np.float64))

        # Should warn when data is made contiguous.
        with self.assertWarns(UserWarning):
            p_data = np.ones((pg.shape[0] * 2, pg.get_num_angles(), pg.shape[1]), dtype=np.float32)
            ts.Data(pg, p_data[::2, ...])
        with self.assertWarns(UserWarning):
            v_data = np.ones((vg.shape[0] * 2, *vg.shape[1:]), dtype=np.float32)
            ts.Data(vg, v_data[::2, ...])

        ts.Data(pg, np.ones(proj_shape, dtype=np.float32))
        ts.Data(vg, np.ones(vg.shape, dtype=np.float32))

    def test_with(self):
        """Test that data in a with statement gets cleaned up

        Also, that using the with statement works.
        """
        seg_fault = False
        pg = ts.cone()
        vg = ts.VolumeGeometry()

        with ts.Data(pg, 0) as pd, ts.Data(vg, 0) as vd:
            proj = pd.get()
            vol = vd.get()

        # No test to check that the code below segfaults, but it does :)
        # You can run the code..
        if seg_fault:
            proj[:] = 0
            vol[:] = 0

    def test_get_set(self):
        """Test data.set() and data.get()
        """

        pg = ts.cone().reshape(10)
        data = ts.Data(pg, 0)

        self.assertTrue(np.all(abs(data.get()) < ts.epsilon))
        data.set(1)
        self.assertTrue(np.all(abs(data.get() - 1) < ts.epsilon))

    def test_is_volume_projection(self):
        self.assertTrue(ts.Data(ts.cone()).is_projection())
        self.assertFalse(ts.Data(ts.cone()).is_volume())
        self.assertTrue(ts.Data(ts.VolumeGeometry()).is_volume())
        self.assertFalse(ts.Data(ts.VolumeGeometry()).is_projection())
