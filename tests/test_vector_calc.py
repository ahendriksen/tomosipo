#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for vector calc."""


import tomosipo as ts
import tomosipo.vector_calc as vc
import numpy as np
import pytest


def test_broadcast_lengths():
    assert vc.broadcast_lengths(1, 1) == 1
    assert vc.broadcast_lengths(1, 5) == 5
    assert vc.broadcast_lengths(5, 1) == 5
    assert vc.broadcast_lengths(3, 3) == 3

    with pytest.raises(ValueError):
        vc.broadcast_lengths(2, 3)

def test_broadcastv():
    x, y = vc._broadcastv(np.ones((3,)), np.ones((3,)))
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)

    x, y = vc._broadcastv(np.ones((1, 3)), np.ones((3,)))
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)

    x, y = vc._broadcastv(np.ones((1, 3)), np.ones((1, 3)))
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)

    x, y = vc._broadcastv(np.ones((3,)), np.ones((1, 3)))
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)

    # More than 2 dims
    with pytest.raises(ValueError):
        x, y = vc._broadcastv(np.ones((1, 1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        x, y = vc._broadcastv(np.ones((1, 3)), np.ones((1, 1, 3)))
    # Non-matching rows
    with pytest.raises(ValueError):
        x, y = vc._broadcastv(np.ones((2, 3)), np.ones((3, 3)))


def test_broadcastmv():
    """Test something."""

    M, x = vc._broadcastmv(np.ones((3, 3)), np.ones(3))
    assert M.shape == (1, 3, 3)
    assert x.shape == (1, 3)

    M, x = vc._broadcastmv(np.ones((1, 3, 3)), np.ones(3))
    assert M.shape == (1, 3, 3)
    assert x.shape == (1, 3)

    M, x = vc._broadcastmv(np.ones((1, 3, 3)), np.ones((1, 3)))
    assert M.shape == (1, 3, 3)
    assert x.shape == (1, 3)

    M, x = vc._broadcastmv(np.ones((2, 3, 3)), np.ones((1, 3)))
    assert M.shape == (2, 3, 3)
    assert x.shape == (2, 3)

    M, x = vc._broadcastmv(np.ones((2, 3, 11)), np.ones((1, 11)))
    assert M.shape == (2, 3, 11)
    assert x.shape == (2, 11)
    assert vc.matrix_transform(M, x).shape == (2, 3)

    with pytest.raises(ValueError):
        # Matrix shape does not match vector length
        vc._broadcastmv(np.ones((2, 3, 11)), np.ones((1, 3)))
    with pytest.raises(ValueError):
        # Different # rows
        M, x = vc._broadcastmv(np.ones((2, 3, 3)), np.ones((3, 3)))


def test_broadcastmm():
    A, B = vc._broadcastmm(np.ones((3, 3)), np.ones((3, 3)))
    assert A.shape == (1, 3, 3)
    assert B.shape == (1, 3, 3)

    A, B = vc._broadcastmm(np.ones((1, 3, 3)), np.ones((3, 3)))
    assert A.shape == (1, 3, 3)
    assert B.shape == (1, 3, 3)

    A, B = vc._broadcastmm(np.ones((1, 3, 3)), np.ones((1, 3, 3)))
    assert A.shape == (1, 3, 3)
    assert B.shape == (1, 3, 3)

    A, B = vc._broadcastmm(np.ones((1, 3, 11)), np.ones((1, 11, 3)))
    assert A.shape == (1, 3, 11)
    assert B.shape == (1, 11, 3)
    vc.matrix_matrix_transform(A, B)

    A, B = vc._broadcastmm(np.ones((1, 2, 11)), np.ones((1, 11, 3)))
    assert A.shape == (1, 2, 11)
    assert B.shape == (1, 11, 3)
    assert vc.matrix_matrix_transform(A, B).shape == (1, 2, 3)

    # Different # rows
    with pytest.raises(ValueError):
        A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((3, 3, 3)))
    # Matrix shapes do not match
    with pytest.raises(ValueError):
        A, B = vc._broadcastmm(np.ones((2, 3, 11)), np.ones((3, 3, 3)))
    with pytest.raises(ValueError):
        A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((3, 11, 3)))
    # More than 3 dims
    with pytest.raises(ValueError):
        A, B = vc._broadcastmm(np.ones((1, 2, 3, 3)), np.ones((3, 11, 3)))
    with pytest.raises(ValueError):
        A, B = vc._broadcastmm(np.ones((2, 3, 3)), np.ones((1, 3, 11, 3)))


def test_orthogonal_basis():
    vecs = vc.to_vec([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
    a0, a1, a2 = vc.orthogonal_basis_from_axis(vecs)

    # TODO: vc.norm should return values in to_scalar format..
    vec_norm = vc.to_scalar(vc.norm(vecs))
    b0 = vc.to_vec((1, 0, 0))
    b1 = vc.to_vec((0, 1, 0))
    b2 = vc.to_vec((0, 0, 1))
    assert round(abs(np.mean(abs(a0 * vec_norm - vecs)) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a0, a1))) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a0, a2))) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a1, a2))) - 0), 7) == 0
    assert np.all(
        abs((vc.cross_product(a1, a2) - a0) - (vc.cross_product(b1, b2) - b0))
        < 10 * ts.epsilon
    )

    # Test random vectors as well:
    vecs = np.random.normal(size=(1000, 3))
    a0, a1, a2 = vc.orthogonal_basis_from_axis(vecs)
    vec_norm = vc.norm(vecs)[:, None]
    assert round(abs(np.mean(abs(a0 * vec_norm - vecs)) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a0, a1))) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a0, a2))) - 0), 7) == 0
    assert round(abs(np.mean(abs(vc.dot(a1, a2))) - 0), 7) == 0
    assert np.all(
        abs((vc.cross_product(a1, a2) - a0) - (vc.cross_product(b1, b2) - b0))
        < 10 * ts.epsilon
    )


def test_cross_product():
    b0 = vc.to_vec((1, 0, 0))
    b1 = vc.to_vec((0, 1, 0))
    b2 = vc.to_vec((0, 0, 1))
    z, y, x = b0, b1, b2

    # Test: X x Y = -Z
    assert round(abs(np.sum(abs(vc.cross_product(x, y) - (-z))) - 0.0), 7) == 0

    # Test: aX x bY = -abZ
    for a, b in np.random.normal(size=(10, 2)):
        c = a * b
        assert (
            round(abs(np.sum(abs(vc.cross_product(a * x, b * y) - (-c * z))) - 0.0), 7)
            == 0
        )
