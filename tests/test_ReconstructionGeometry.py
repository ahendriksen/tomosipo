#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ReconstructionGeometry."""


import unittest
import numpy as np
import astra
import tomosipo as ts


class TestReconstructionGeometry(unittest.TestCase):
    """Tests for ReconstructionGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_forward_backward(self):
        pd = ts.Data(ts.cone().reshape(10))
        vd = ts.Data(ts.VolumeGeometry().reshape(10))

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
        interactive = False
        p = ts.Data(ts.cone(angles=100).reshape(100))
        v = ts.Data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

        # Fill the projection data with random noise:
        proj = p.get()
        proj[:] = np.random.normal(size=proj.shape)
        proj[:] = abs(proj)

        if interactive:
            ts.display_data(p)

        ts.fdk(v, p)

        if interactive:
            ts.display_data(v)
