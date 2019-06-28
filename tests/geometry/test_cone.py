#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ConeGeometry."""

import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform, random_cone
import tomosipo.vector_calc as vc


###############################################################################
#                             Creation and dunders                            #
###############################################################################
def test_cone():
    assert ts.geometry.cone.ConeGeometry() == ts.cone()


def test_init():
    """Test init."""
    interactive = False
    pg = ts.cone()

    assert isinstance(pg, ts.geometry.base_projection.ProjectionGeometry)

    pg = ts.cone(angles=np.linspace(0, 1, 100))
    assert pg.num_angles == 100

    with pytest.raises(ValueError):
        pg = ts.cone(angles="asdf")
    with pytest.raises(ValueError):
        pg = ts.cone(angles=[])

    representation = repr(pg)
    if interactive:
        print(ts.cone())
        print(representation)


def test_equal():
    """Test __eq__

    Make sure that a ConeGeometry is equal to it
    """

    pg = ts.cone()
    unequal = [
        ts.cone(angles=2),
        ts.cone(size=5),
        ts.cone(shape=3),
        ts.cone(src_obj_dist=50),
        ts.cone(src_det_dist=50),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert pg != u


def test_get_item():
    pg = ts.cone(angles=10, shape=20)
    assert pg[1].num_angles == 1
    assert pg[:1].num_angles == 1
    assert pg[:2].num_angles == 2
    assert pg[:].num_angles == 10
    assert pg[-1] == pg[9]
    assert pg[-2] == pg[8]

    with pytest.raises(ValueError):
        pg[10]


###############################################################################
#                                     to_*                                    #
###############################################################################
def test_to_vec():
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

        s_dst = np.random.uniform(10, 100)
        d_dst = np.random.uniform(0, 100)

        pg1 = ts.cone(angles, size, shape, d_dst, s_dst)
        pgv = pg1.to_vec()
        assert pg1.det_shape == pgv.det_shape


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

        s_dst = np.random.uniform(10, 100)
        d_dst = np.random.uniform(0, 100)

        pg1 = ts.cone(angles, size, shape, d_dst, s_dst)
        astra_pg = pg1.to_astra()
        pg2 = ts.from_astra_geometry(astra_pg)

        assert pg1 == pg2


###############################################################################
#                                  Properties                                 #
###############################################################################


def test_det_sizes():
    size = (1, 2)
    pg = ts.cone(angles=3, size=size)
    size = np.array(size)[None, ...]
    assert abs(pg.det_sizes - size).sum() < ts.epsilon


def test_src_obj_det_dist():
    assert ts.cone(src_obj_dist=5.0).src_obj_dist == 5.0
    assert ts.cone(src_det_dist=3.0).src_det_dist == 3.0


###############################################################################
#                                   Methods                                   #
###############################################################################
def test_rescale_det():
    pg = ts.cone(angles=10, shape=20)

    assert approx(pg.det_sizes) == pg.rescale_det(2).det_sizes
    assert pg.rescale_det((2, 2)) == pg.rescale_det(2)
    assert pg.rescale_det(10).det_shape == (2, 2)


def test_transform():
    for _ in range(10):
        pg = random_cone()
        T1 = random_transform()
        T2 = random_transform()

        with pytest.warns(Warning):
            assert T1(T2)(pg) == T1(T2(pg))
            assert ts.identity()(pg) == pg.to_vec()


def test_to_box():
    pg = ts.cone(10, shape=(5, 3), src_obj_dist=11, src_det_dist=21)
    src_box, det_box = pg.to_box()

    assert pytest.approx(0) == vc.norm(src_box.pos - det_box.pos) - (10 + 11)
