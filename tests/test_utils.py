#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for utils."""


import unittest
from tomosipo.utils import up_tuple


class Testutils(unittest.TestCase):
    """Tests for utils."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_up_tuple(self):
        """Test up_tuple."""

        self.assertEqual((1, 1), up_tuple((1, 1), 2))
        self.assertEqual((1, 2), up_tuple((1, 2), 2))
        self.assertEqual((1, 1), up_tuple([1], 2))
        self.assertEqual((0, 0), up_tuple(range(1), 2))
        self.assertEqual((0, 0), up_tuple([0], 2))
        self.assertEqual((0, 0), up_tuple((0,), 2))
        self.assertEqual((0, 0), up_tuple(iter((0,)), 2))
