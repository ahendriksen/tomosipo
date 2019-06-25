#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ProjectionGeometry."""

import pytest
import tomosipo as ts
from tomosipo.ProjectionGeometry import ProjectionGeometry, is_projection_geometry


@pytest.fixture
def default_proj_geoms():
    return [
        ts.cone(),
        ts.cone().to_vec(),
        ts.cone(angles=10, shape=(5, 3)),
        ts.cone(angles=10, shape=(5, 3)).to_vec(),
        ts.cone(angles=11, shape=(10, 10)),
        ts.cone(angles=11, shape=(10, 10)).to_vec(),
    ]


def test_is_projection_geometry():
    assert is_projection_geometry(ts.cone())
    assert not is_projection_geometry(ts.volume())
    assert not is_projection_geometry(None)


def test_init():
    """Test init."""
    pg = ProjectionGeometry()
    with pytest.raises(ValueError):
        ProjectionGeometry(0)

    assert pg.det_shape == (1, 1)


def test_interface(default_proj_geoms):
    for pg in default_proj_geoms:
        assert is_projection_geometry(pg)
        repr(pg)
        assert pg == pg
        pg_fix = ts.from_astra_geometry(pg.to_astra())
        assert pg_fix == ts.from_astra_geometry(pg_fix.to_astra())
        assert pg.to_vec() == pg.to_vec().to_vec()
        assert pg.is_cone or pg.is_parallel
        pg.is_vec
        pg.num_angles
        if pg.is_cone:
            assert pg.src_pos.shape == (pg.num_angles, 3)
        if pg.is_parallel:
            assert pg.ray_dir.shape == (pg.num_angles, 3)

        assert pg.det_pos.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_u.shape == (pg.num_angles, 3)
        assert pg.det_sizes.shape == (pg.num_angles, 2)
        assert pg.corners.shape == (pg.num_angles, 4, 3)
        if pg.det_shape[0] % 2 == 0 and pg.det_shape[1] % 2 == 0:
            assert pg.rescale_det(2).det_shape == (
                pg.det_shape[0] // 2,
                pg.det_shape[1] // 2,
            )

        assert pg.reshape(1).det_shape == (1, 1)
        pg.project_point((0, 0, 0))
