#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ProjectionGeometry."""


import unittest
import numpy as np
import tomosipo as ts


class TestProjectionGeometry(unittest.TestCase):
    """Tests for ProjectionGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_is_projection_geometry(self):
        self.assertTrue(ts.is_projection_geometry(ts.cone()))
        self.assertFalse(ts.is_projection_geometry(ts.VolumeGeometry()))
        self.assertFalse(ts.is_projection_geometry(None))

    def test_init(self):
        """Test init."""
        pg = ts.ProjectionGeometry()
        with self.assertRaises(ValueError):
            ts.ProjectionGeometry(0)

        self.assertEqual(pg.shape, (1, 1))
