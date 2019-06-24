#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for utils."""


import unittest
from tomosipo.utils import up_tuple, index_one_dim


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

    def test_one_dim(self):
        (l, r, length, ps) = index_one_dim(0, 1, 1, 0)
        self.assertEqual((l, r, length, ps), (0, 1, 1, 1))

        (l, r, length, ps) = index_one_dim(0, 1, 1, slice(None, None, None))
        self.assertEqual((l, r, length, ps), (0, 1, 1, 1))

        (l, r, length, ps) = index_one_dim(0, 0, 1, 0)
        self.assertEqual((l, r, length, ps), (0, 0, 1, 0))

        (l, r, length, ps) = index_one_dim(0, 0, 0, 0)
        self.assertEqual((l, r, length, ps), (0, 0, 0, 1))
