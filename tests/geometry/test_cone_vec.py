#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ConeVectorGeometry."""

from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform, random_cone_vec
import tomosipo.vector_calc as vc
from tomosipo.geometry import transform
import pytest


def test_init():
    """Test something."""
    interactive = False

    # This should not raise an exception
    vecs = np.arange(30).reshape((10, 3))
    kwargs = dict(src_pos=vecs, det_pos=vecs, det_v=vecs, det_u=vecs)
    pg = ts.cone_vec(shape=1, **kwargs)

    representation = repr(pg)
    if interactive:
        print(representation)

    with pytest.raises(TypeError):
        ts.cone_vec(0, **kwargs)  # missing shape=
    with pytest.raises(TypeError):
        ts.cone_vec(shape=0, **kwargs)  # shape cannot be zero
    with pytest.raises(TypeError):
        two_dim = np.arange(20).reshape((10, 2))
        ts.cone_vec(
            shape=1, src_pos=two_dim, det_pos=two_dim, det_v=two_dim, det_u=two_dim
        )
    with pytest.raises(ValueError):
        diff_shaped_args = dict(
            src_pos=np.random.normal(size=(10, 3)),
            det_pos=np.random.normal(size=(11, 3)),
            det_v=np.random.normal(size=(12, 3)),
            det_u=np.random.normal(size=(13, 3)),
        )
        ts.cone_vec(shape=1, **diff_shaped_args)


def test_repr():
    x = [[1, 2, 3], [4, 5, 6]]
    pg = ts.cone_vec(shape=10, src_pos=x, det_pos=x, det_v=x, det_u=x)
    r = """ts.cone_vec(
    shape=(10, 10),
    src_pos=array([[1., 2., 3.],
       [4., 5., 6.]]),
    det_pos=array([[1., 2., 3.],
       [4., 5., 6.]]),
    det_v=array([[1., 2., 3.],
       [4., 5., 6.]]),
    det_u=array([[1., 2., 3.],
       [4., 5., 6.]]),
)"""

    assert repr(pg) == r
    with np.printoptions(legacy="1.13"):
        assert repr(pg) == r

    pg = ts.geometry.random_cone_vec()
    assert eval(repr(pg), dict(ts=ts, array=np.array)) == pg


def test_equal():
    """Test __eq__

    Make sure that a ConeVectorGeometry is equal to itself.
    """
    x = np.array([(0, 0, 0)])
    pg = ts.cone_vec(shape=10, src_pos=x, det_pos=x, det_v=x, det_u=x)
    unequal = [
        ts.cone_vec(shape=9, src_pos=x, det_pos=x, det_v=x, det_u=x),
        ts.cone_vec(shape=10, src_pos=x + 1, det_pos=x, det_v=x, det_u=x),
        ts.cone_vec(shape=10, src_pos=x, det_pos=x + 1, det_v=x, det_u=x),
        ts.cone_vec(shape=10, src_pos=x, det_pos=x, det_v=x + 1, det_u=x),
        ts.cone_vec(shape=10, src_pos=x, det_pos=x, det_v=x, det_u=x + 1),
        ts.cone(angles=2, cone_angle=1 / 2),
        ts.cone(size=np.sqrt(2), cone_angle=1 / 2),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert pg != u


def test_getitem():
    pg = ts.cone(angles=10, shape=100, size=np.sqrt(2), cone_angle=1 / 2).to_vec()

    # Test indexing with boolean arrays
    assert pg[np.ones(pg.num_angles) == 1] == pg
    assert pg[np.arange(pg.num_angles) % 2 == 0] == pg[0::2]

    assert pg == pg[:, :, :]

    assert 2 * pg[::2].num_angles == pg.num_angles
    assert 10 * pg[::10].num_angles == pg.num_angles

    assert round(abs(0.0 - np.sum(abs(pg[:, ::2].det_v - 2 * pg.det_v))), 7) == 0
    assert round(abs(0.0 - np.sum(abs(pg[:, :, ::2].det_u - 2 * pg.det_u))), 7) == 0

    assert round(abs(0.0 - np.sum(abs(pg[:, :, ::2].det_pos - pg.det_pos))), 7) != 0
    assert (
        round(abs(0.0 - np.sum(abs(pg[:, :, ::2].det_pos - pg[:, :, 1::2].det_pos))), 7)
        != 0
    )
    assert (
        round(abs(0.0 - np.sum(abs(pg[:, :, ::2].det_v - pg[:, :, 1::2].det_v))), 7)
        == 0
    )
    assert (
        round(abs(0.0 - np.sum(abs(pg[:, :, ::2].det_u - pg[:, :, 1::2].det_u))), 7)
        == 0
    )
    assert (
        round(
            abs(0.0 - np.sum(abs(pg[:, :, ::2].det_sizes - pg[:, :, 1::2].det_sizes))),
            7,
        )
        == 0
    )


def test_to_from_astra():
    nrows = 30
    num_tests = 100
    for _ in range(num_tests):
        shape = tuple(int(np.random.uniform(1, 100)) for _ in range(2))
        v1, v2, v3, v4 = (np.random.uniform(size=(nrows, 3)) for _ in range(4))

        pg = ts.cone_vec(shape=shape, src_pos=v1, det_pos=v2, det_v=v3, det_u=v4)
        astra_pg = pg.to_astra()
        assert pg == ts.from_astra(astra_pg)


def test_project_point_detector_spacing():
    """Test project_point with detector spacing

    Test whether projection_point works with non-uniform detector spacing.
    """
    shape = (10, 40)
    size = (30, 80)

    pg = ts.cone(
        angles=1, shape=shape, size=size, src_orig_dist=10, src_det_dist=10
    ).to_vec()
    assert round(abs(np.abs(pg.project_point((0, 0, 0))).sum() - 0), 7) == 0

    assert (
        round(abs(np.abs(pg.project_point((3.0, 0, 0)) - [1.0, 0.0]).sum() - 0), 7) == 0
    )
    assert (
        round(abs(np.abs(pg.project_point((0, 0, 2.0)) - [0.0, 1.0]).sum() - 0), 7) == 0
    )

    shape = (100, 400)
    pg = ts.cone(
        angles=1, shape=shape, size=size, src_orig_dist=10, src_det_dist=10
    ).to_vec()
    assert round(abs(np.abs(pg.project_point((0, 0, 0))).sum() - 0), 7) == 0
    assert (
        round(abs(np.abs(pg.project_point((0.3, 0, 0)) - [1.0, 0.0]).sum() - 0), 7) == 0
    )
    assert (
        round(abs(np.abs(pg.project_point((0, 0, 0.2)) - [0.0, 1.0]).sum() - 0), 7) == 0
    )


def test_det_sizes():
    size = (1, 1)
    pg = ts.cone(angles=5, size=size, cone_angle=1 / 2).to_vec()
    assert abs(size - pg.det_sizes).sum() < ts.epsilon
    for _ in range(10):
        new_shape = np.random.uniform(1, 100, size=2).astype(np.int)
        # Ensure that reshape does not increase the detector size.
        pg2 = pg.reshape(new_shape)
        assert pg2.det_shape == tuple(new_shape)
        assert round(abs(0.0 - np.sum(abs(pg2.det_sizes - pg.det_sizes))), 7) == 0


def test_corners():
    # TODO: This test deserves better..
    size = (1, 1)
    pg1 = ts.cone(angles=5, size=size, cone_angle=1 / 2).to_vec()

    assert pg1.corners.shape == (5, 4, 3)


def test_src_pos():
    pg1 = ts.cone(size=np.sqrt(2), src_orig_dist=1, src_det_dist=1).to_vec()
    pg2 = ts.cone(size=np.sqrt(2), src_orig_dist=2, src_det_dist=2).to_vec()

    source_diff = pg1.src_pos * 2 - pg2.src_pos
    assert np.all(abs(source_diff) < ts.epsilon)


def test_transform():

    for _ in range(10):
        pg = random_cone_vec()
        T1 = random_transform()
        T2 = random_transform()

        assert (T1 * T2) * pg == T1 * (T2 * pg)
        assert transform.identity() * pg == pg


def test_to_vol():

    pg = ts.cone(
        angles=10, shape=(5, 3), size=np.sqrt(2), src_orig_dist=11, src_det_dist=21
    ).to_vec()

    assert np.allclose(pg.det_pos, pg.to_vol().pos)
    # XXX: Really do not know what to test here..


def test_rescale_det():
    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20).to_vec()

    assert (
        round(abs(0.0 - np.sum(abs(pg.det_sizes - pg.rescale_det(2).det_sizes))), 7)
        == 0
    )

    assert pg.rescale_det((2, 2)) == pg.rescale_det(2)
    assert pg.rescale_det(10).det_shape == (2, 2)


def test_project_point():
    for _ in range(5):
        pg = ts.geometry.random_cone_vec()
        # The detector position should always be projected on (0, 0)
        assert pg.project_point(pg.det_pos) == approx(np.zeros((pg.num_angles, 2)))
        # A translation of the projection geometry along the ray_dir
        # should not affect project_point.
        p = vc.to_vec((3, 5, 7))
        T = ts.translate(np.random.normal() * (pg.src_pos - p))
        assert pg.project_point(p).shape == (T * pg).project_point(p).shape
        assert pg.project_point(p) == approx((T * pg).project_point(p))
