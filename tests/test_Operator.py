#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for operator."""


import unittest
import numpy as np
import tomosipo as ts


interactive = False


class TestOperator(unittest.TestCase):
    """Tests for ReconstructionGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_forward_backward(self):
        pd = ts.data(ts.cone(shape=10))
        vd = ts.data(ts.volume(10))

        rs = [
            ([pd, vd], {}),
            ([pd, vd], dict(detector_supersampling=2, voxel_supersampling=2)),
            ((pd, vd), dict(detector_supersampling=1, voxel_supersampling=2)),
            ((pd, vd), dict(detector_supersampling=2, voxel_supersampling=1)),
        ]

        for data, kwargs in rs:
            ts.forward(*data, **kwargs)
            ts.backward(*data, **kwargs)

    def test_fdk(self):
        p = ts.data(ts.cone(angles=100).reshape(100))
        v = ts.data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

        # Fill the projection data with random noise:
        p.data[:] = np.random.normal(size=p.data.shape)
        p.data[:] = abs(p.data)

        if interactive:
            ts.display_data(p)

        ts.fdk(v, p)

        if interactive:
            ts.display_data(v)

    def test_operator(self):
        pg = ts.cone(angles=150, shape=100)
        vg = ts.volume(shape=100)

        A = ts.operator(vg, pg, additive=False)
        x = ts.phantom.hollow_box(ts.data(vg))

        y = A(x)
        y_ = A(np.copy(x.data))  # Test with np.array input

        self.assertAlmostEqual(0.0, np.sum(abs(y.data - y_.data)))

        # Test with `Data` and `np.array` again:
        x1 = A.transpose(y)
        x2 = A.transpose(y.data)
        self.assertAlmostEqual(0.0, np.sum(abs(x1.data - x2.data)))

    def test_operator_additive(self):
        pg = ts.cone(angles=150, shape=100)
        vg = ts.volume(shape=100)

        A = ts.operator(vg, pg, additive=False)
        B = ts.operator(vg, pg, additive=True)

        x = ts.phantom.hollow_box(ts.data(vg))
        y = ts.data(pg)

        B(x, out=y)
        B(x, out=y)

        self.assertAlmostEqual(0.0, np.mean(abs(2 * A(x).data - y.data)))
