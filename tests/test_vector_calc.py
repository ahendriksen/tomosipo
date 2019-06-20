#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for vector calc."""


import unittest
import tomosipo as ts
import tomosipo.vector_calc as vc
import numpy as np


class Testvector_calc(unittest.TestCase):
    """Tests for vector calc."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_broadcastv(self):
        x, y = vc._broadcastv(np.ones((3,)), np.ones((3,)))
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(y.shape, (1, 3))

        x, y = vc._broadcastv(np.ones((1, 3)), np.ones((3,)))
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(y.shape, (1, 3))

        x, y = vc._broadcastv(np.ones((1, 3)), np.ones((1, 3)))
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(y.shape, (1, 3))

        x, y = vc._broadcastv(np.ones((3,)), np.ones((1, 3)))
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(y.shape, (1, 3))

        # More than 2 dims
        with self.assertRaises(ValueError):
            x, y = vc._broadcastv(np.ones((1, 1, 3)), np.ones((1, 3)))
        with self.assertRaises(ValueError):
            x, y = vc._broadcastv(np.ones((1, 3)), np.ones((1, 1, 3)))
        # Non-matching rows
        with self.assertRaises(ValueError):
            x, y = vc._broadcastv(np.ones((2, 3)), np.ones((3, 3)))

    def test_broadcastmv(self):
        """Test something."""

        M, x = vc._broadcastmv(np.ones((3, 3)), np.ones(3))
        self.assertEqual(M.shape, (1, 3, 3))
        self.assertEqual(x.shape, (1, 3))

        M, x = vc._broadcastmv(np.ones((1, 3, 3)), np.ones(3))
        self.assertEqual(M.shape, (1, 3, 3))
        self.assertEqual(x.shape, (1, 3))

        M, x = vc._broadcastmv(np.ones((1, 3, 3)), np.ones((1, 3)))
        self.assertEqual(M.shape, (1, 3, 3))
        self.assertEqual(x.shape, (1, 3))

        M, x = vc._broadcastmv(np.ones((2, 3, 3)), np.ones((1, 3)))
        self.assertEqual(M.shape, (2, 3, 3))
        self.assertEqual(x.shape, (2, 3))

        M, x = vc._broadcastmv(np.ones((2, 3, 11)), np.ones((1, 11)))
        self.assertEqual(M.shape, (2, 3, 11))
        self.assertEqual(x.shape, (2, 11))
        self.assertEqual(vc.matrix_transform(M, x).shape, (2, 3))

        with self.assertRaises(ValueError):
            # Matrix shape does not match vector length
            vc._broadcastmv(np.ones((2, 3, 11)), np.ones((1, 3)))
        with self.assertRaises(ValueError):
            # Different # rows
            M, x = vc._broadcastmv(np.ones((2, 3, 3)), np.ones((3, 3)))

    def test_broadcastmm(self):
        A, B = vc._broadcastmm(np.ones((3, 3)), np.ones((3, 3)))
        self.assertEqual(A.shape, (1, 3, 3))
        self.assertEqual(B.shape, (1, 3, 3))

        A, B = vc._broadcastmm(np.ones((1, 3, 3)), np.ones((3, 3)))
        self.assertEqual(A.shape, (1, 3, 3))
        self.assertEqual(B.shape, (1, 3, 3))

        A, B = vc._broadcastmm(np.ones((1, 3, 3)), np.ones((1, 3, 3)))
        self.assertEqual(A.shape, (1, 3, 3))
        self.assertEqual(B.shape, (1, 3, 3))

        A, B = vc._broadcastmm(np.ones((1, 3, 11)), np.ones((1, 11, 3)))
        self.assertEqual(A.shape, (1, 3, 11))
        self.assertEqual(B.shape, (1, 11, 3))
        vc.matrix_matrix_transform(A, B)

        A, B = vc._broadcastmm(np.ones((1, 2, 11)), np.ones((1, 11, 3)))
        self.assertEqual(A.shape, (1, 2, 11))
        self.assertEqual(B.shape, (1, 11, 3))
        self.assertEqual(vc.matrix_matrix_transform(A, B).shape, (1, 2, 3))

        # Different # rows
        with self.assertRaises(ValueError):
            A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((3, 3, 3)))
        # Matrix shapes do not match
        with self.assertRaises(ValueError):
            A, B = vc._broadcastmm(np.ones((2, 3, 11)), np.ones((3, 3, 3)))
        with self.assertRaises(ValueError):
            A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((3, 11, 3)))
        # More than 3 dims
        with self.assertRaises(ValueError):
            A, B = vc._broadcastmm(np.ones((1, 2, 3, 3)), np.ones((3, 11, 3)))
        with self.assertRaises(ValueError):
            A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((1, 3, 11, 3)))

    def test_orthogonal_basis(self):
        vecs = vc.to_vec([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
        a0, a1, a2 = vc.orthogonal_basis_from_axis(vecs)

        # TODO: vc.norm should return values in to_scalar format..
        vec_norm = vc.to_scalar(vc.norm(vecs))
        b0 = vc.to_vec((1, 0, 0))
        b1 = vc.to_vec((0, 1, 0))
        b2 = vc.to_vec((0, 0, 1))
        self.assertAlmostEqual(np.mean(abs(a0 * vec_norm - vecs)), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a0, a1))), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a0, a2))), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a1, a2))), 0)
        self.assertTrue(
            np.all(
                abs((vc.cross_product(a1, a2) - a0) - (vc.cross_product(b1, b2) - b0))
                < 10 * ts.epsilon
            )
        )

        # Test random vectors as well:
        vecs = np.random.normal(size=(1000, 3))
        a0, a1, a2 = vc.orthogonal_basis_from_axis(vecs)
        vec_norm = vc.norm(vecs)[:, None]
        self.assertAlmostEqual(np.mean(abs(a0 * vec_norm - vecs)), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a0, a1))), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a0, a2))), 0)
        self.assertAlmostEqual(np.mean(abs(vc.dot(a1, a2))), 0)
        self.assertTrue(
            np.all(
                abs((vc.cross_product(a1, a2) - a0) - (vc.cross_product(b1, b2) - b0))
                < 10 * ts.epsilon
            )
        )

    def test_cross_product(self):
        b0 = vc.to_vec((1, 0, 0))
        b1 = vc.to_vec((0, 1, 0))
        b2 = vc.to_vec((0, 0, 1))
        z, y, x = b0, b1, b2

        # Test: X x Y = -Z
        self.assertAlmostEqual(np.sum(abs(vc.cross_product(x, y) - (-z))), 0.0)

        # Test: aX x bY = -abZ
        for a, b in np.random.normal(size=(10, 2)):
            c = a * b
            self.assertAlmostEqual(
                np.sum(abs(vc.cross_product(a * x, b * y) - (-c * z))), 0.0
            )
