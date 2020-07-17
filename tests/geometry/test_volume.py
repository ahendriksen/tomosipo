#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for VolumeGeometry."""


import pytest
from pytest import approx
import numpy as np
import itertools
import tomosipo as ts
from tomosipo.geometry import random_volume, random_transform
from tomosipo.geometry.volume import (
    from_astra,
    _pos_size_to_extent,
    _extent_to_pos_size,
)
from tomosipo.geometry import transform
from .test_transform import translations, scalings


def test_pos_size_and_extent():
    N = 10
    for _ in range(N):
        pos = np.random.normal(size=3)
        size = np.square(np.random.normal(size=3))
        extent = _pos_size_to_extent(pos, size)
        newpos, newsize = _extent_to_pos_size(extent)

        assert np.allclose(newpos, pos)
        assert np.allclose(newsize, size)


def test_is_volume():
    assert ts.geometry.is_volume(ts.volume())
    assert ts.geometry.is_volume(ts.volume().to_vec())
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
    assert ts.volume(pos=(0.5, 0.5, 0.5), size=1).extent == extent

    size = (1, 1, 1)
    pos = 0
    # All 8 possible combinations of providing and not providing pos,
    # size, and extent. Three of them should raise an error.
    with pytest.raises(ValueError):
        ts.volume(pos=pos, size=size, extent=extent)
    ts.volume(pos=pos, size=size, extent=None)
    with pytest.raises(ValueError):
        ts.volume(pos=pos, size=None, extent=extent)
    ts.volume(pos=pos, size=None, extent=None)
    with pytest.raises(ValueError):
        ts.volume(pos=None, size=size, extent=extent)
    ts.volume(pos=None, size=size, extent=None)
    ts.volume(pos=None, size=None, extent=extent)
    ts.volume(pos=None, size=None, extent=None)

    # Malformed extent parameter
    with pytest.raises(TypeError):
        ts.volume(extent=(1, 0))
    with pytest.raises(TypeError):
        ts.volume(extent=3)


def test_equal():
    """Test __eq__

    """
    vg = ts.volume()
    unequal = [ts.volume(shape=10), ts.volume(shape=(10, 9, 8)), ts.cone()]

    assert vg == vg
    for u in unequal:
        assert vg != u


def test_repr():
    # TODO: Use ts.volume when reparametrized
    unit_box = ts.geometry.VolumeGeometry(
        shape=(1, 2, 3), pos=(4, 5, 6), size=(7, 8, 9)
    )
    r = """VolumeGeometry(
    shape=(1, 2, 3),
    pos=(4.0, 5.0, 6.0),
    size=(7.0, 8.0, 9.0),
)"""

    assert repr(unit_box) == r


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

    print(vg)
    print(vg[:, :, :])

    assert vg == vg[:, :, :]
    assert vg == vg[:, :]
    assert vg == vg[:]

    assert vg[0, 0, 0] == vg[:1, :1, :1]
    assert vg[0] == vg[0, :, :]
    assert vg[-1] == vg[2]

    assert vg[::3, ::4, ::5].shape == (1, 1, 1)
    # Check that voxel size is preserved:
    assert np.allclose(vg[0, 1, 2].size, (np.array(vg.size) / vg.shape))

    assert np.allclose(vg.size[1], vg[:, ::2, :].size[1])
    assert np.allclose(vg.shape[1], 2 * vg[:, ::2].shape[1])

    assert np.allclose(
        2 * vg.size[1] / vg.shape[1], vg[:, ::2, :].size[1] / vg[:, ::2, :].shape[1],
    )

    assert np.allclose(vg[:, ::2, :].size, vg[:, 1::2, :].size)


@pytest.mark.parametrize("T, S", itertools.product(translations, scalings))
def test_properties_under_transformations(T, S):
    vg = ts.geometry.random_volume()
    P = ts.from_perspective(box=vg)
    TS = T * S
    assert (TS * vg)[0, 0, 0] == TS * vg[0, 0, 0]
    assert (TS * vg)[-1, -1, -1] == TS * vg[-1, -1, -1]
    assert (S * P * vg).pos == approx((P * vg).pos)
    assert (T * vg).w == approx(vg.w)
    assert (T * vg).v == approx(vg.v)
    assert (T * vg).u == approx(vg.u)
    assert (TS * vg).shape == vg.shape
    assert (T * vg).size == vg.size
    assert (T * vg).sizes == approx(vg.sizes)
    assert (T * vg).voxel_sizes == approx(vg.voxel_sizes)
    assert (T * vg).voxel_size == vg.voxel_size
    # corners?
    # lowerleftcorner?


@pytest.mark.parametrize("T, S", itertools.product(translations, scalings))
def test_translation_scaling(T, S):
    """Test translation and scaling

    VolumeGeometry implements scaling and translation directly. It
    makes sense to check that performing translation and scaling works
    the same in vec and non-vec geometries.

    """
    vg = ts.geometry.random_volume()
    assert (T * vg).to_vec() == T * vg.to_vec()
    print(S)
    print(S * vg)
    print((S * vg).to_vec())
    print(S * vg.to_vec())
    print(vg.to_vec())
    assert (S * vg).to_vec() == S * vg.to_vec()
    assert ((S * T) * vg).to_vec() == (S * T) * vg.to_vec()
    assert ((T * S) * vg).to_vec() == (T * S) * vg.to_vec()


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
    t = (5, 5, 5)
    assert vg in vg
    assert not (vg in vg.translate(t))
    assert not (vg in vg.scale(0.5))
    assert vg in vg.scale(2)
    assert vg.translate(t) in vg.translate(t).scale(2)


def test_reshape():
    vg = ts.volume()

    vg1 = vg.reshape(100)
    vg2 = vg.reshape((100,) * 3)

    assert vg1 == vg2


def test_to_vec():
    vg = ts.volume(shape=(3, 5, 7))
    vg_vec = vg.to_vec()
    assert np.allclose(vg_vec.size, vg.size)


def test_with_voxel_size():
    vg = ts.volume(shape=10, size=10, pos=0)
    assert vg.with_voxel_size(1.0) == vg
    assert vg.with_voxel_size(2.0) == ts.volume(shape=5, size=10, pos=0)

    assert vg.with_voxel_size(1.0) == vg.with_voxel_size((1.0, 1.0, 1.0))
    assert vg.with_voxel_size(2.0) == vg.with_voxel_size((2.0, 2.0, 2.0))
    assert vg.with_voxel_size(3.0) == vg.with_voxel_size((3.0, 3.0, 3.0))
