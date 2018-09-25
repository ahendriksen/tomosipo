#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for display."""


import unittest
import tomosipo as ts
import numpy as np


class Testdisplay(unittest.TestCase):
    """Tests for display."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_display_data(self):
        interactive = False
        p = ts.Data(ts.cone(angles=100).reshape(100))
        v = ts.Data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

        # Fill the projection data with random noise:
        proj = p.get()
        proj[:] = np.random.normal(size=proj.shape)
        proj[:] = abs(proj)

        r = ts.ReconstructionGeometry(p, v)

        if interactive:
            ts.display_data(p)
        r.backward()
        if interactive:
            ts.display_data(v)

    def test_display_geometry(self):
        """Test something."""
        interactive = False
        pg = ts.cone(angles=100, source_distance=10, detector_distance=5)
        vg = ts.volume_from_projection_geometry(pg, inside=False)

        if interactive:
            ts.display_geometry(pg, vg)
