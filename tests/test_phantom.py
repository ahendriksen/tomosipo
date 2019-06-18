#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for phantom generation."""


import unittest
import tomosipo as ts


class Testphantom(unittest.TestCase):
    """Tests for phantom generation."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_hollow_box(self):
        """Test something."""

        for s in [10, 20, 30]:
            vd = ts.data(ts.volume(shape=100))
            ts.phantom.hollow_box(vd)
            self.assertAlmostEqual(vd.data.mean(), 0.208)
