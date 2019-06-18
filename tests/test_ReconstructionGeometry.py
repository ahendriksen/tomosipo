#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ReconstructionGeometry."""


import unittest
import numpy as np
import tomosipo as ts

interactive = False


class TestReconstructionGeometry(unittest.TestCase):
    """Tests for ReconstructionGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_forward_backward(self):
        pd = ts.data(ts.cone().reshape(10))
        vd = ts.data(ts.VolumeGeometry().reshape(10))

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
