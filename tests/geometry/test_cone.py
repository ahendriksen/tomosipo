#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ConeGeometry."""

import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform, random_cone
from tomosipo.geometry import transform

###############################################################################
#                             Creation and dunders                            #
###############################################################################


def test_init(interactive):
    """Test init."""

    pg = ts.cone(cone_angle=1)

    # Check shape and angles
    with pytest.raises(TypeError):
        ts.cone(angles=1, shape=None, cone_angle=1)
    with pytest.raises(TypeError):
        assert ts.cone(angles=None, shape=1, cone_angle=1).num_angles == 1

    # Check combinations of cone_angle, src_obj_dist, and src_det_dist.
    ass_kws = dict(angles=1, shape=1, size=1)
    with pytest.raises(ValueError):
        ts.cone(**ass_kws, cone_angle=1, src_obj_dist=1, src_det_dist=1)
    with pytest.raises(ValueError):
        ts.cone(**ass_kws, cone_angle=1, src_obj_dist=1, src_det_dist=None)
    with pytest.raises(ValueError):
        ts.cone(**ass_kws, cone_angle=1, src_obj_dist=None, src_det_dist=1)
    with pytest.raises(ValueError):
        ts.cone(**ass_kws, cone_angle=None, src_obj_dist=None, src_det_dist=None)
    # These should all be fine:
    ts.cone(**ass_kws, cone_angle=1, src_obj_dist=None, src_det_dist=None)
    ts.cone(**ass_kws, cone_angle=None, src_obj_dist=1, src_det_dist=1)
    ts.cone(**ass_kws, cone_angle=None, src_obj_dist=1, src_det_dist=None)
    ts.cone(**ass_kws, cone_angle=None, src_obj_dist=None, src_det_dist=1)

    # Check that size equals shape if not given:
    assert ts.cone(shape=2, cone_angle=1).det_size == (2, 2)
    assert ts.cone(shape=2, size=1, cone_angle=1).det_size == (1, 1)

    assert isinstance(pg, ts.geometry.base_projection.ProjectionGeometry)
    pg = ts.cone(angles=np.linspace(0, 1, 100), cone_angle=1)
    assert pg.num_angles == 100

    with pytest.raises(ValueError):
        pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=[])

    representation = repr(pg)
    if interactive:
        print(ts.cone(size=np.sqrt(2), cone_angle=1 / 2))
        print(representation)


def test_repr():
    pg = ts.cone(angles=10, shape=11, size=1, src_obj_dist=5, src_det_dist=6)
    r = """ts.cone(
    angles=10,
    shape=(11, 11),
    size=(1, 1),
    src_obj_dist=5.0,
    src_det_dist=6.0,
)"""

    assert repr(pg) == r
    with np.printoptions(legacy="1.13"):
        assert repr(pg) == r

    pg = ts.geometry.random_cone()
    assert eval(repr(pg), dict(ts=ts, array=np.array)) == pg


def test_equal():
    """Test __eq__

    Make sure that a ConeGeometry is equal to it
    """

    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2)
    unequal = [
        ts.cone(size=5, cone_angle=1 / 2),
        ts.cone(angles=2, size=np.sqrt(2), cone_angle=1 / 2),
        ts.cone(shape=2, size=np.sqrt(2), cone_angle=1 / 2),
        ts.cone(size=np.sqrt(2), src_obj_dist=50),
        ts.cone(size=np.sqrt(2), src_det_dist=50),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert pg != u


def test_get_item():
    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)
    assert pg[1].num_angles == 1
    assert pg[:1].num_angles == 1
    assert pg[-1].num_angles == 1
    assert pg[:2].num_angles == 2
    assert pg[:].num_angles == 10
    assert pg[-1] == pg[9]
    assert pg[-2] == pg[8]

    assert pg[np.ones(pg.num_angles) == 1] == pg
    assert pg[np.arange(pg.num_angles) % 2 == 0] == pg[0::2]

    with pytest.raises(IndexError):
        ts.cone(angles=3, cone_angle=1)[4]

    with pytest.raises(ValueError):
        # Indexing on the detector plane is not supported.
        pg[:, 0, 0]


###############################################################################
#                                     to_*                                    #
###############################################################################
def test_to_vec():
    num_tests = 10
    for _ in range(num_tests):
        num_angles = int(np.random.uniform(1, 100))

        tupled = np.random.uniform() < 0.5
        if tupled:
            angles = np.random.uniform(0, 2 * np.pi, num_angles)
        else:
            angles = num_angles

        if tupled:
            shape = np.random.uniform(1, 100, size=2).astype(np.int)
            size = np.random.uniform(1, 100, size=2)
        else:
            shape = np.random.uniform(1, 100, size=1).astype(np.int)
            size = np.random.uniform(1, 100, size=1)

        pg = ts.cone(
            angles=angles,
            shape=shape,
            size=size,
            src_obj_dist=np.random.uniform(10, 100),
            src_det_dist=np.random.uniform(0, 100),
        )
        assert pg.det_shape == pg.to_vec().det_shape


def test_to_from_astra():
    """Test to and from astra conversion functions

    This implementation checks the to and from astra conversion
    functions for cone beam geometries. It generates `num_tests`
    test cases and checks if the conversion to and from astra does
    not change the projection geometry.
    """
    num_tests = 100
    for _ in range(num_tests):
        num_angles = int(np.random.uniform(1, 100))

        tupled = np.random.uniform() < 0.5
        if tupled:
            angles = np.random.uniform(0, 2 * np.pi, num_angles)
        else:
            angles = num_angles

        if tupled:
            shape = np.random.uniform(1, 100, size=2).astype(np.int)
            size = np.random.uniform(1, 100, size=2)
        else:
            shape = np.random.uniform(1, 100, size=1).astype(np.int)
            size = np.random.uniform(1, 100, size=1)

        pg = ts.cone(
            angles=angles, shape=shape, size=size, cone_angle=np.random.uniform(0.5, 1)
        )

        assert pg == ts.from_astra(pg.to_astra())


###############################################################################
#                                  Properties                                 #
###############################################################################


def test_det_sizes():
    size = (1, 2)
    pg = ts.cone(angles=3, cone_angle=1 / 2, size=size)
    size = np.array(size)[None, ...]
    assert abs(pg.det_sizes - size).sum() < ts.epsilon


def test_src_obj_det_dist():
    assert ts.cone(size=np.sqrt(2), src_obj_dist=5.0).src_obj_dist == 5.0
    assert ts.cone(size=np.sqrt(2), src_det_dist=3.0).src_det_dist == 3.0


###############################################################################
#                                   Methods                                   #
###############################################################################
def test_rescale_det():
    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)

    assert approx(pg.det_sizes) == pg.rescale_det(2).det_sizes
    assert pg.rescale_det((2, 2)) == pg.rescale_det(2)
    assert pg.rescale_det(10).det_shape == (2, 2)


def test_transform():
    for _ in range(10):
        pg = random_cone()
        T1 = random_transform()
        T2 = random_transform()

        with pytest.warns(Warning):
            assert (T1 * T2) * pg == T1 * (T2 * pg)
            assert transform.identity() * pg == pg.to_vec()


def test_to_box():
    pg = ts.cone(
        angles=10, shape=(5, 3), size=np.sqrt(2), src_obj_dist=11, src_det_dist=21
    )
    assert pg.det_pos == approx(pg.to_box().pos)
