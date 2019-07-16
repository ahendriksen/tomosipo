import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform
import tomosipo.geometry.det_vec as dv
from tomosipo.utils import up_tuple
from tomosipo.geometry import transform


@pytest.fixture
def det_vecs():
    p, v, u = [(0, 0, 0)], [(1, 0, 0)], [(0, 0, 1)]
    num_angles = 100
    return [
        dv.det_vec((1, 1), num_angles * p, num_angles * v, num_angles * u),
        dv.det_vec(100, num_angles * p, num_angles * v, num_angles * u),
        dv.random_det_vec(),
        dv.random_det_vec(),
        dv.random_det_vec(),
        dv.random_det_vec(),
    ]


def test_init():
    vecs = np.arange(30).reshape((10, 3))
    assert dv.det_vec(1, vecs, vecs, vecs).det_shape == (1, 1)
    assert dv.det_vec((3, 5), vecs, vecs, vecs).det_shape == (3, 5)

    with pytest.raises(ValueError):
        # Shape of 0 is not allowed
        dv.det_vec(0, vecs, vecs, vecs)

    with pytest.raises(ValueError):
        vecs = np.random.normal(size=(10, 2))
        dv.det_vec(1, vecs, vecs, vecs)


def test_equal():
    pos = np.array([(0, 0, 0)])
    pg = dv.det_vec(10, pos, pos, pos)
    unequal = [
        dv.det_vec(9, pos, pos, pos),
        dv.det_vec(10, pos + 1, pos, pos),
        dv.det_vec(10, pos, pos + 1, pos),
        dv.det_vec(10, pos, pos, pos + 1),
        ts.cone(angles=2),
        ts.cone(),
        ts.cone().to_vec(),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert u != pg
        assert pg != u


def test_getitem(det_vecs):
    for pg in det_vecs:
        assert pg == pg[:]
        assert pg == pg[:, :]
        assert pg == pg[:, :, :]

        assert pg[1::2].num_angles == pg.num_angles // 2
        assert pg[9::10].num_angles == pg.num_angles // 10

        if (
            pg.num_angles % 2 != 0
            or pg.det_shape[0] % 2 != 0
            or pg.det_shape[1] % 2 != 0
        ):
            continue

        assert pg[:, ::2].det_v == approx(2 * pg.det_v)
        assert pg[:, ::2].det_pos != approx(pg.det_pos)
        assert pg[:, :, ::2].det_u == approx(2 * pg.det_u)
        assert pg[:, :, ::2].det_pos != approx(pg.det_pos)

        assert pg[:, ::2].det_pos != approx(pg[:, 1::2].det_pos)
        assert pg[:, :, ::2].det_pos != approx(pg[:, :, 1::2].det_pos)

        assert pg[:, ::2, :].det_v == approx(pg[:, 1::2, :].det_v)
        assert pg[:, ::2, :].det_u == approx(pg[:, 1::2, :].det_u)
        assert pg[:, :, ::2].det_v == approx(pg[:, :, 1::2].det_v)
        assert pg[:, :, ::2].det_u == approx(pg[:, :, 1::2].det_u)
        assert pg[:, ::2, :].det_sizes == approx(pg[:, 1::2, :].det_sizes)
        assert pg[:, :, ::2].det_sizes == approx(pg[:, :, 1::2].det_sizes)


def test_astra(det_vecs):
    for pg in det_vecs:
        astra_pg = pg.to_astra()
        assert pg == dv.DetectorVectorGeometry.from_astra(astra_pg)


def test_to_vec(det_vecs):
    for pg in det_vecs:
        assert pg == pg.to_vec()


def test_to_box(det_vecs):
    with pytest.warns(Warning):
        # Should for non-uniform detector sizes
        for pg in det_vecs:
            assert pg.det_pos == approx(pg.to_box().pos)

    # TODO: This test deserves better..


def test_src_pos(det_vecs):
    for pg in det_vecs:
        with pytest.raises(NotImplementedError):
            pg.src_pos


def test_ray_dir(det_vecs):
    for pg in det_vecs:
        with pytest.raises(NotImplementedError):
            pg.ray_dir


def test_det_properties(det_vecs):
    for pg in det_vecs:
        assert pg.det_pos.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_sizes.shape == (pg.num_angles, 2)
        assert pg.corners.shape == (pg.num_angles, 4, 3)


def test_det_normal(det_vecs):
    unit_det = ts.geometry.det_vec.det_vec(
        shape=1, det_pos=(0, 0, 0), det_v=(1, 0, 0), det_u=(0, 0, 1)
    )
    assert unit_det.det_normal == approx(np.array([(0, 1, 0)]))


def test_corners(det_vecs):
    for pg in det_vecs:
        assert pg.lower_left_corner == approx(pg.corners[:, 0, :])
        assert pg.det_pos - pg.lower_left_corner == approx(
            pg.det_shape[0] * pg.det_v / 2 + pg.det_shape[1] * pg.det_u / 2
        )


def test_reshape(det_vecs):
    for pg in det_vecs:
        for shape in [1, (3, 5)]:
            pg_reshaped = pg.reshape(shape)
            assert pg_reshaped.det_shape == up_tuple(shape, 2)
            assert pg_reshaped.det_sizes == approx(pg.det_sizes)


def test_rescale_det_vecs(det_vecs):
    for pg in det_vecs:
        if pg.det_shape[0] % 2 == 0 and pg.det_shape[1] % 2 == 0:
            assert pg.rescale_det(2).det_shape[0] == pg.det_shape[0] // 2
            assert pg.rescale_det(2).det_shape[1] == pg.det_shape[1] // 2
            assert pg.rescale_det(2).det_sizes == approx(pg.det_sizes)
            assert pg.rescale_det(2).det_v == approx(2 * pg.det_v)
            assert pg.rescale_det(2).det_u == approx(2 * pg.det_u)


def test_transform(det_vecs):
    for pg in det_vecs:
        T1 = random_transform()
        T2 = random_transform()

        assert (T1 * T2) * pg == T1 * (T2 * pg)
        assert transform.identity() * pg == pg


def test_project_point(det_vecs):
    for pg in det_vecs:
        with pytest.raises(NotImplementedError):
            pg.project_point((0, 0, 0))
