#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for VolumeGeometry."""


import pytest
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_volume, random_transform
from tomosipo.geometry.volume import from_astra
from tomosipo.geometry import transform


def test_is_volume():
    assert ts.geometry.is_volume(ts.volume())
    assert not ts.geometry.is_volume(ts.cone())
    assert not ts.geometry.is_volume(None)


def test_init():
    """Test init."""
    vg = ts.volume()
    assert vg.shape == (1, 1, 1)
    # Check shape
    assert ts.volume(shape=(3, 4, 5)).shape == (3, 4, 5)
    # Check extent
    extent = ((0, 1), (3, 4), (5, 6))
    assert ts.volume(extent=extent).extent == extent
    extent = ((0, 1),) * 3
    assert ts.volume(extent=(0, 1)).extent == extent

    assert ts.volume(center=0.5, size=1).extent == extent

    # Check errors
    with pytest.raises(ValueError):
        ts.volume(extent=(1, 0))
    with pytest.raises(TypeError):
        ts.volume(extent=3)
    with pytest.raises(ValueError):
        ts.volume(center=0, size=None)
    with pytest.raises(ValueError):
        ts.volume(center=None, size=1)
    with pytest.raises(ValueError):
        ts.volume(center=0, size=1, extent=(0, 1))


def test_equal():
    """Test __eq__

    """
    vg = ts.volume()
    unequal = [ts.volume(shape=10), ts.volume(shape=(10, 9, 8)), ts.cone()]

    assert vg == vg
    for u in unequal:
        assert vg != u


def test_volume():
    assert ts.volume() == ts.volume()
    shapes = [2, (1, 4, 5), (10, 10, 10)]
    for s in shapes:
        assert ts.volume(s) == ts.volume(s)


def test_astra():
    vg = ts.volume()
    vg = vg.scale((1, 2, 3)).translate((10, 20, 30))
    vg1 = from_astra(vg.to_astra())

    assert vg == vg1


def test_getitem():
    vg = ts.volume(shape=(3, 4, 5), extent=((11, 13), (17, 19), (23, 29)))

    assert vg == vg[:, :, :]
    assert vg == vg[:, :]
    assert vg == vg[:]

    assert vg[0, 0, 0] == vg[:1, :1, :1]
    assert vg[0] == vg[0, :, :]
    assert vg[-1] == vg[2]

    assert vg[::3, ::4, ::5].shape == (1, 1, 1)
    # Check that voxel size is preserved:
    assert np.allclose(vg[0, 1, 2].size(), (np.array(vg.size()) / vg.shape))

    assert np.allclose(vg.size()[1], vg[:, ::2, :].size()[1])
    assert np.allclose(vg.shape[1], 2 * vg[:, ::2].shape[1])

    assert np.allclose(
        2 * vg.size()[1] / vg.shape[1],
        vg[:, ::2, :].size()[1] / vg[:, ::2, :].shape[1],
    )

    assert np.allclose(vg[:, ::2, :].size(), vg[:, 1::2, :].size())


def test_translate():
    vg = ts.volume()

    assert vg == vg.translate((0, 0, 0))
    assert vg == vg.translate(0)
    assert vg == vg.untranslate((0, 0, 0))
    assert vg == vg.untranslate(0)
    for _ in range(10):
        t = np.random.normal(size=3)
        assert vg == vg.translate(t).untranslate(t)
        # Apply transformation or call method directly:
        assert ts.translate(t) * vg == vg.translate(t)


def test_scale():
    vg = ts.volume()

    assert vg == vg.scale((1, 1, 1))
    assert vg == vg.scale(1)
    for _ in range(10):
        t = abs(np.random.normal(size=3)) + 0.01
        assert vg == vg.scale(t).scale(1 / t)
        # Apply transformation or call method directly:
        assert ts.scale(t) * vg == vg.scale(t)


def test_transform():
    for _ in range(10):
        vg = random_volume()
        T1 = random_transform()
        T2 = random_transform()
        # Check that random rotations trigger a warning that the
        # volume has been converted to a vector geometry.
        with pytest.warns(UserWarning):
            T1 * T2 * vg
        assert (T1 * T2) * vg.to_vec() == T1 * (T2 * vg.to_vec())
        assert transform.identity() * vg.to_vec() == T1.inv * (T1 * vg.to_vec())


def test_contains():
    vg = ts.volume()
    assert not (vg in vg.translate(1))
    assert not (vg in vg.scale(0.5))
    assert vg in vg.scale(2)
    assert vg.translate(5) in vg.translate(5).scale(2)
    assert vg in vg


def test_intersection():
    vg = ts.volume()

    for _ in range(100):
        t = np.random.normal(size=3)
        r = np.random.normal(size=3)
        r = abs(r) + 0.01

        vg2 = vg.translate(t).scale(r)

        intersection = vg.intersect(vg2)

        if intersection is not None:
            assert intersection in vg
            assert intersection in vg2
        else:
            assert not (vg in vg2)
            assert not (vg2 in vg)

        assert vg2 == vg2.intersect(vg2)


def test_reshape():
    vg = ts.volume()

    vg1 = vg.reshape(100)
    vg2 = vg.reshape((100,) * 3)

    assert vg1 == vg2


def test_to_vec():
    vg = ts.volume(shape=(3, 5, 7))
    vg_vec = vg.to_vec()
    assert np.allclose(vg_vec.size, vg.size())


def test_with_voxel_size():
    vg = ts.volume(shape=10, size=10, center=0)
    assert vg.with_voxel_size(1.0) == vg
    assert vg.with_voxel_size(2.0) == ts.volume(shape=5, size=10, center=0)

    assert vg.with_voxel_size(1.0) == vg.with_voxel_size((1.0, 1.0, 1.0))
    assert vg.with_voxel_size(2.0) == vg.with_voxel_size((2.0, 2.0, 2.0))
    assert vg.with_voxel_size(3.0) == vg.with_voxel_size((3.0, 3.0, 3.0))
