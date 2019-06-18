#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for display."""


import unittest
import tomosipo as ts

interactive = False


class Testdisplay(unittest.TestCase):
    """Tests for display."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_display_data(self):
        p = ts.data(ts.cone(angles=100, shape=100))
        v = ts.data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

        # Fill v with hollow box phantom
        ts.phantom.hollow_box(v)
        ts.forward(v, p)

        if interactive:
            ts.display(v)
        if interactive:
            ts.display(p)

    def test_display_geometry(self):
        """Test something."""
        pg = ts.cone(angles=100, source_distance=10, detector_distance=5)
        vg = ts.volume_from_projection_geometry(pg, inside=False)

        if interactive:
            ts.display(pg, vg)
