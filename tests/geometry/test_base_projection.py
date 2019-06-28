#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ProjectionGeometry."""


import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry.base_projection import ProjectionGeometry, is_projection
import tomosipo.vector_calc as vc


@pytest.fixture
def default_proj_geoms():
    return [
        ts.cone(),
        ts.cone().to_vec(),
        ts.cone(angles=10, shape=(5, 3)),
        ts.cone(angles=10, shape=(5, 3)).to_vec(),
        ts.cone(angles=11, shape=(10, 10)),
        ts.cone(angles=11, shape=(10, 10)).to_vec(),
        ts.geometry.det_vec.random_det_vec(),
        ts.geometry.random_parallel_vec(),
        ts.geometry.random_parallel(),
    ]


def test_is_projection(default_proj_geoms):
    for pg in default_proj_geoms:
        assert is_projection(pg)

    assert not is_projection(ts.volume())
    assert not is_projection(None)


def test_init():
    """Test init."""
    pg = ProjectionGeometry()
    with pytest.raises(ValueError):
        ProjectionGeometry(0)

    assert pg.det_shape == (1, 1)


def test_interface(default_proj_geoms):
    for pg in default_proj_geoms:
        assert is_projection(pg)
        repr(pg)
        assert pg == pg
        pg_fix = ts.from_astra_geometry(pg.to_astra())
        assert pg_fix == ts.from_astra_geometry(pg_fix.to_astra())
        assert pg.to_vec() == pg.to_vec().to_vec()
        pg.is_vec
        pg.num_angles
        if pg.is_vec:
            with pytest.raises(NotImplementedError):
                pg.angles
        else:
            assert len(pg.angles) == pg.num_angles

        if pg.is_cone:
            assert pg.src_pos.shape == (pg.num_angles, 3)
        if pg.is_parallel:
            assert pg.ray_dir.shape == (pg.num_angles, 3)

        assert pg.det_pos.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_u.shape == (pg.num_angles, 3)

        assert vc.cross_product(pg.det_u, pg.det_v) == approx(pg.det_normal)

        if pg.is_vec and pg.num_angles == 1:
            assert pg.det_sizes[0] == approx(pg.det_size)
        if pg.is_vec and pg.num_angles > 1:
            with pytest.raises(ValueError):
                # If the detector size is constant, check that it is
                # consistent.
                assert pg.det_sizes[0] == approx(pg.det_size)
                # If so, make the detector size differ:
                S = ts.scale(np.random.normal(size=(pg.num_angles, 3)))
                # The detector size should not be uniform anymore, and
                # this should raise an error.
                S(pg).det_size
        else:
            assert len(pg.det_size) == 2

        assert pg.det_sizes.shape == (pg.num_angles, 2)
        assert pg.corners.shape == (pg.num_angles, 4, 3)
        if pg.det_shape[0] % 2 == 0 and pg.det_shape[1] % 2 == 0:
            assert pg.rescale_det(2).det_shape == (
                pg.det_shape[0] // 2,
                pg.det_shape[1] // 2,
            )

        assert pg.reshape(1).det_shape == (1, 1)
        if pg.is_parallel or pg.is_cone:
            pg.project_point((0, 0, 0))
