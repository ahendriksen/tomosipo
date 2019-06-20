#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ProjectionGeometry."""


import unittest
import numpy as np
import tomosipo as ts
from tomosipo.ProjectionGeometry import ProjectionGeometry, is_projection_geometry


class TestProjectionGeometry(unittest.TestCase):
    """Tests for ProjectionGeometry."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_is_projection_geometry(self):
        self.assertTrue(is_projection_geometry(ts.cone()))
        self.assertFalse(is_projection_geometry(ts.VolumeGeometry()))
        self.assertFalse(is_projection_geometry(None))

    def test_init(self):
        """Test init."""
        pg = ProjectionGeometry()
        with self.assertRaises(ValueError):
            ProjectionGeometry(0)

        self.assertEqual(pg.shape, (1, 1))
