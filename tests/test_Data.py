#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Data."""


import numpy as np
import tomosipo as ts
import pytest


def test_init():
    """Test data creation."""
    pg = ts.cone(angles=10, shape=20, cone_angle=1 / 2)
    vg = ts.volume().reshape(10)
    pd = ts.data(pg, 0)

    proj_shape = pd.data.shape

    # Should warn when data is silently converted to float32.
    with pytest.warns(UserWarning):
        ts.data(pg, np.ones(proj_shape, dtype=np.float64))
    with pytest.warns(UserWarning):
        ts.data(vg, np.ones(vg.shape, dtype=np.float64))

    # Should warn when data is made contiguous.
    with pytest.warns(UserWarning):
        p_data = np.ones(
            (pg.det_shape[0] * 2, pg.num_angles, pg.det_shape[1]), dtype=np.float32
        )
        ts.data(pg, p_data[::2, ...])
    with pytest.warns(UserWarning):
        v_data = np.ones((vg.shape[0] * 2, *vg.shape[1:]), dtype=np.float32)
        ts.data(vg, v_data[::2, ...])

    # Should raise an error when geometry is not convertible to ASTRA:
    with pytest.raises(TypeError):
        ts.data(vg.to_vec())

    ts.data(pg, np.ones(proj_shape, dtype=np.float32))
    ts.data(vg, np.ones(vg.shape, dtype=np.float32))


def test_with():
    """Test that data in a with statement gets cleaned up

    Also, that using the with statement works.
    """
    pg = ts.cone(cone_angle=1 / 2)
    vg = ts.volume()

    with ts.data(pg, 0) as pd, ts.data(vg, 0) as vd:
        proj = pd.data
        vol = vd.data


def test_data():
    """Test data.data property"""

    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2).reshape(10)
    d = ts.data(pg, 0)

    assert np.all(abs(d.data) < ts.epsilon)
    d.data[:] = 1.0
    assert np.all(abs(d.data - 1) < ts.epsilon)


def test_is_volume_projection():
    assert ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2)).is_projection()
    assert not ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2)).is_volume()
    assert ts.data(ts.volume()).is_volume()
    assert not ts.data(ts.volume()).is_projection()


def test_init_idempotency():
    """Test that ts.data can be used idempotently

    In numpy, you do `np.array(np.array([1, 1]))' to cast a list
    to a numpy array. The second invocation of `np.array'
    basically short-circuits and returns the argument.

    Here, we test that we have the same behaviour for `ts.data'.
    """
    vg = ts.volume(shape=10)

    vd = ts.data(vg)
    vd_ = ts.data(vg, vd)

    assert id(vd) == id(vd_)

    with pytest.raises(ValueError):
        ts.data(ts.volume(), vd)

    pg = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)
    # TODO: Implement and use .copy()
    pg_ = ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=10, shape=20)

    pd = ts.data(pg)
    pd_ = ts.data(pg_, pd)

    assert id(pd) == id(pd_)

    with pytest.raises(ValueError):
        ts.data(vg, pd)

    with pytest.raises(ValueError):
        ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2), pd)
