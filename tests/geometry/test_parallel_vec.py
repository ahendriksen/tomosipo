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
            shape=(1, 1),
            ray_dir=num_angles * r,
            det_pos=num_angles * p,
            det_v=num_angles * v,
            det_u=num_angles * u,
        ),
        ts.parallel_vec(
            shape=100,
            ray_dir=num_angles * r,
            det_pos=num_angles * p,
            det_v=num_angles * v,
            det_u=num_angles * u,
        ),
        random_parallel_vec(),
        random_parallel_vec(),
        random_parallel_vec(),
        random_parallel_vec(),
    ]


def test_init():
    vecs = np.arange(30).reshape((10, 3))
    kwargs = dict(ray_dir=vecs, det_pos=vecs, det_v=vecs, det_u=vecs,)
    assert ts.parallel_vec(shape=1, **kwargs).det_shape == (1, 1)
    assert ts.parallel_vec(shape=(3, 5), **kwargs).det_shape == (3, 5)

    with pytest.raises(TypeError):
        # must use `shape=1`:
        ts.parallel_vec(1, **kwargs)

    with pytest.raises(ValueError):
        # Shape of 0 is not allowed
        ts.parallel_vec(shape=0, **kwargs)

    with pytest.raises(ValueError):
        vecs = np.random.normal(size=(10, 2))
        two_dim_args = dict(ray_dir=vecs, det_pos=vecs, det_v=vecs, det_u=vecs,)
        ts.parallel_vec(shape=1, **two_dim_args)

    with pytest.raises(ValueError):
        diff_shaped_args = dict(
            ray_dir=np.random.normal(size=(10, 3)),
            det_pos=np.random.normal(size=(11, 3)),
            det_v=np.random.normal(size=(12, 3)),
            det_u=np.random.normal(size=(13, 3)),
        )
        ts.parallel_vec(shape=1, **diff_shaped_args)


def test_equal():
    x = np.array([(0, 0, 0)])
    pg = ts.parallel_vec(shape=10, ray_dir=x, det_pos=x, det_v=x, det_u=x)
    unequal = [
        ts.parallel_vec(shape=9, ray_dir=x, det_pos=x, det_v=x, det_u=x),
        ts.parallel_vec(shape=10, ray_dir=x + 1, det_pos=x, det_v=x, det_u=x),
        ts.parallel_vec(shape=10, ray_dir=x, det_pos=x + 1, det_v=x, det_u=x),
        ts.parallel_vec(shape=10, ray_dir=x, det_pos=x, det_v=x + 1, det_u=x),
        ts.parallel_vec(shape=10, ray_dir=x, det_pos=x, det_v=x, det_u=x + 1),
        ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=2),
        ts.cone(size=np.sqrt(2), cone_angle=1 / 2),
        ts.cone(size=np.sqrt(2), cone_angle=1 / 2).to_vec(),
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
