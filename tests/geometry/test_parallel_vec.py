import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform, random_parallel_vec
from tomosipo.utils import up_tuple
import tomosipo.vector_calc as vc
from tomosipo.geometry import transform


@pytest.fixture
def par_vecs():
    r, p, v, u = [(0, 1, 0)], [(0, 0, 0)], [(1, 0, 0)], [(0, 0, 1)]
    num_angles = 100
    return [
        ts.parallel_vec(
            (1, 1), num_angles * r, num_angles * p, num_angles * v, num_angles * u
        ),
        ts.parallel_vec(
            100, num_angles * r, num_angles * p, num_angles * v, num_angles * u
        ),
        random_parallel_vec(),
        random_parallel_vec(),
        random_parallel_vec(),
        random_parallel_vec(),
    ]


def test_init():
    vecs = np.arange(30).reshape((10, 3))
    assert ts.parallel_vec(1, vecs, vecs, vecs, vecs).det_shape == (1, 1)
    assert ts.parallel_vec((3, 5), vecs, vecs, vecs, vecs).det_shape == (3, 5)

    with pytest.raises(ValueError):
        # Shape of 0 is not allowed
        ts.parallel_vec(0, vecs, vecs, vecs, vecs)

    with pytest.raises(ValueError):
        vecs = np.random.normal(size=(10, 2))
        ts.parallel_vec(1, vecs, vecs, vecs, vecs)


def test_equal():
    x = np.array([(0, 0, 0)])
    pg = ts.parallel_vec(10, x, x, x, x)
    unequal = [
        ts.parallel_vec(9, x, x, x, x),
        ts.parallel_vec(10, x + 1, x, x, x),
        ts.parallel_vec(10, x, x + 1, x, x),
        ts.parallel_vec(10, x, x, x + 1, x),
        ts.parallel_vec(10, x, x, x + 1, x),
        ts.cone(angles=2),
        ts.cone(),
        ts.cone().to_vec(),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert u != pg
        assert pg != u


def test_getitem(par_vecs):
    for pg in par_vecs:
        assert pg == pg[:]
        assert pg == pg[:, :]
        assert pg == pg[:, :, :]

        assert pg[1::2].num_angles == pg.num_angles // 2
        assert pg[9::10].num_angles == pg.num_angles // 10

        assert pg[1:9:2].ray_dir == approx(pg.ray_dir[1:9:2, :])
        assert pg[-1].ray_dir.shape == pg.ray_dir[-1:, :].shape
        assert pg[-1].ray_dir == approx(pg.ray_dir[-1:, :])
        assert pg[-2].ray_dir.shape == pg.ray_dir[-2:-1:, :].shape
        assert pg[-2].ray_dir == approx(pg.ray_dir[-2:-1:, :])

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


def test_astra(par_vecs):
    for pg in par_vecs:
        astra_pg = pg.to_astra()
        assert pg == ts.from_astra(astra_pg)


def test_to_vec(par_vecs):
    for pg in par_vecs:
        assert pg == pg.to_vec()


def test_to_box(par_vecs):
    for pg in par_vecs:
        assert pg.det_pos == approx(pg.to_box().pos)
    # TODO: This test deserves better..


def test_src_pos(par_vecs):
    for pg in par_vecs:
        with pytest.raises(NotImplementedError):
            pg.src_pos


def test_det_properties(par_vecs):
    for pg in par_vecs:
        assert pg.ray_dir.shape == (pg.num_angles, 3)
        assert pg.det_pos.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_sizes.shape == (pg.num_angles, 2)
        assert pg.corners.shape == (pg.num_angles, 4, 3)


def test_corners(par_vecs):
    for pg in par_vecs:
        assert pg.lower_left_corner == approx(pg.corners[:, 0, :])
        assert pg.det_pos - pg.lower_left_corner == approx(
            pg.det_shape[0] * pg.det_v / 2 + pg.det_shape[1] * pg.det_u / 2
        )


def test_reshape(par_vecs):
    for pg in par_vecs:
        for shape in [1, (3, 5)]:
            pg_reshaped = pg.reshape(shape)
            assert pg_reshaped.ray_dir == approx(pg.ray_dir)
            assert pg_reshaped.det_shape == up_tuple(shape, 2)
            assert pg_reshaped.det_sizes == approx(pg.det_sizes)


def test_rescale_par_vecs(par_vecs):
    for pg in par_vecs:
        if pg.det_shape[0] % 2 == 0 and pg.det_shape[1] % 2 == 0:
            assert pg.rescale_det(2).ray_dir == approx(pg.ray_dir)
            assert pg.rescale_det(2).det_shape[0] == pg.det_shape[0] // 2
            assert pg.rescale_det(2).det_shape[1] == pg.det_shape[1] // 2
            assert pg.rescale_det(2).det_sizes == approx(pg.det_sizes)
            assert pg.rescale_det(2).det_v == approx(2 * pg.det_v)
            assert pg.rescale_det(2).det_u == approx(2 * pg.det_u)


def test_transform(par_vecs):
    for pg in par_vecs:
        T1 = random_transform()
        T2 = random_transform()

        assert (T1 * T2) * pg == T1 * (T2 * pg)
        assert transform.identity() * pg == pg

    for pg in par_vecs:
        T = ts.translate(np.random.normal(size=3))
        assert (T * pg).ray_dir == approx(pg.ray_dir)


def test_project_point(par_vecs):
    for pg in par_vecs:
        # The detector position should always be projected on (0, 0)
        assert pg.project_point(pg.det_pos) == approx(np.zeros((pg.num_angles, 2)))
        # A translation of the projection geometry along the ray_dir
        # should not affect project_point.
        p = vc.to_vec((3, 5, 7))
        T = ts.translate(pg.ray_dir)
        assert pg.project_point(p).shape == (T * pg).project_point(p).shape
        assert pg.project_point(p) == approx((T * pg).project_point(p))
