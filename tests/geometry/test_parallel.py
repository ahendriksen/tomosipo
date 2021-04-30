import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
from tomosipo.geometry import random_transform, random_parallel
import tomosipo.vector_calc as vc


@pytest.fixture
def par_geoms():
    return [
        ts.parallel(angles=2),
        ts.parallel(angles=10, size=3, shape=11),
        ts.parallel(angles=5, size=7, shape=23),
        random_parallel(),
        random_parallel(),
        random_parallel(),
    ]


def test_init():
    assert ts.parallel(angles=1, shape=1).det_shape == (1, 1)
    assert ts.parallel(angles=1, shape=1).num_angles == 1
    assert ts.parallel(angles=1, size=1).det_sizes == approx(np.ones((1, 2)))
    assert ts.parallel(shape=(3, 5)).det_shape == (3, 5)

    assert ts.parallel(shape=(2, 3)).det_size == (2, 3)
    assert ts.parallel().det_shape == (1, 1)

    # Check that 180° angular arc is created by default.
    angles = np.linspace(0, np.pi, 10, endpoint=False)
    assert ts.parallel(angles=10) == ts.parallel(angles=angles)

    with pytest.raises(TypeError):
        # Shape of 0 is not allowed
        ts.parallel(shape=0)

    with pytest.raises(TypeError):
        # Do not allow empty angles:
        ts.parallel(angles=0)
    with pytest.raises(TypeError):
        # Do not allow 2-dimensional angle arrays
        ts.parallel(angles=np.ones((1, 1)))


def test_repr():
    pg = ts.parallel(angles=10, shape=11, size=1)
    r = """ts.parallel(
    angles=10,
    shape=(11, 11),
    size=(1.0, 1.0),
)"""

    assert repr(pg) == r
    with np.printoptions(legacy="1.13"):
        assert repr(pg) == r

    pg = ts.geometry.random_parallel()
    assert eval(repr(pg), dict(ts=ts, array=np.array)) == pg


def test_equal():
    pg = ts.parallel()
    unequal = [
        ts.parallel(angles=2),
        ts.parallel(size=5),
        ts.parallel(shape=3),
        ts.volume(),
    ]

    assert pg == pg

    for u in unequal:
        assert u != pg
        assert pg != u


def test_getitem(par_geoms):
    pg = ts.parallel(angles=10, shape=20)
    assert pg[1].num_angles == 1
    assert pg[:1].num_angles == 1
    assert pg[-1].num_angles == 1
    assert pg[:2].num_angles == 2
    assert pg[:].num_angles == 10
    assert pg[-1] == pg[9]
    assert pg[-2] == pg[8]

    assert pg[np.ones(pg.num_angles) == 1] == pg
    assert pg[np.arange(pg.num_angles) % 2 == 0] == pg[0::2]

    with pytest.raises(ValueError):
        # Indexing on the detector plane is not supported.
        pg[:, 0, 0]

    with pytest.raises(IndexError):
        ts.parallel(angles=3)[4]

    for pg in par_geoms:
        assert pg == pg[:]

        with pytest.raises(ValueError):
            assert pg == pg[:, :]
        with pytest.raises(ValueError):
            assert pg == pg[:, :, :]

        assert pg[1::2].num_angles == pg.num_angles // 2
        assert pg[1:9:2].angles == approx(pg.angles[1:9:2])
        assert pg[-1].angles.shape == pg.angles[-1:].shape
        assert pg[-1].angles == approx(pg.angles[-1:])
        assert pg[-2].angles.shape == pg.angles[-2:-1:].shape
        assert pg[-2].angles == approx(pg.angles[-2:-1:])


def test_astra(par_geoms):
    for pg in par_geoms:
        astra_pg = pg.to_astra()
        assert pg == ts.from_astra(astra_pg)
        # check that the following yields the same result:
        # 1) parallel -> to_vec -> to_astra -> from_astra
        # 2) parallel -> to_astra -> from_astra -> to_vec
        assert (
            ts.from_astra(pg.to_vec().to_astra())
            == ts.from_astra(pg.to_astra()).to_vec()
        )


def test_to_vec(par_geoms):
    for pg in par_geoms:
        assert pg.det_shape == pg.to_vec().det_shape
        assert pg.num_angles == pg.to_vec().num_angles
        assert pg.det_sizes == approx(pg.to_vec().det_sizes)


def test_to_box(par_geoms):
    for pg in par_geoms:
        assert pg.det_pos == approx(pg.to_box().pos)

    # TODO: This test deserves better..


def test_src_pos(par_geoms):
    for pg in par_geoms:
        with pytest.raises(NotImplementedError):
            pg.src_pos


def test_det_properties(par_geoms):
    for pg in par_geoms:
        assert pg.ray_dir.shape == (pg.num_angles, 3)
        assert pg.det_pos.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_v.shape == (pg.num_angles, 3)
        assert pg.det_sizes.shape == (pg.num_angles, 2)
        assert pg.corners.shape == (pg.num_angles, 4, 3)


def test_corners(par_geoms):
    for pg in par_geoms:
        assert pg.lower_left_corner == approx(pg.corners[:, 0, :])
        assert pg.det_pos - pg.lower_left_corner == approx(
            pg.det_shape[0] * pg.det_v / 2 + pg.det_shape[1] * pg.det_u / 2
        )


def test_reshape(par_geoms):
    for pg in par_geoms:
        pg.reshape(1).det_shape == (1, 1)
        pg.reshape((3, 5)).det_shape == (3, 5)

        pg.reshape(1).det_sizes == pg.det_sizes
        pg.reshape((3, 5)).det_sizes == approx(pg.det_sizes)


def test_rescale_par_geoms(par_geoms):
    for pg in par_geoms:
        if pg.det_shape[0] % 2 == 0 and pg.det_shape[1] % 2 == 0:
            assert pg.rescale_det(2).angles == approx(pg.angles)
            assert pg.rescale_det(2).det_shape[0] == pg.det_shape[0] // 2
            assert pg.rescale_det(2).det_shape[1] == pg.det_shape[1] // 2
            assert pg.rescale_det(2).det_sizes == approx(pg.det_sizes)
            assert pg.rescale_det(2).det_v == approx(2 * pg.det_v)
            assert pg.rescale_det(2).det_u == approx(2 * pg.det_u)


def test_transform(par_geoms):
    for pg in par_geoms:
        with pytest.warns(Warning):
            # Translating the parallel geometry converts it to a
            # parallel vector geometry behind the scenes, for which we
            # get a warning.
            T = random_transform()
            T * pg


def test_project_point(par_geoms):
    for pg in par_geoms:
        # The detector position should always be projected on (0, 0)
        assert pg.project_point(pg.det_pos) == approx(np.zeros((pg.num_angles, 2)))
        # A translation of the projection geometry along the ray_dir
        # should not affect project_point.
        p = vc.to_vec((3, 5, 7))
        T = ts.translate(pg.ray_dir)
        with pytest.warns(UserWarning):
            # Translating the parallel geometry converts it to a
            # parallel vector geometry behind the scenes, for which we
            # get a warning.
            assert pg.project_point(p).shape == (T * pg).project_point(p).shape
            assert pg.project_point(p) == approx((T * pg).project_point(p))
            # ray_dir is orthogonal to the detector plane, so we should
            # also be able to move along the detector normal:
            T = ts.translate(pg.det_normal)
            assert pg.project_point(p).shape == (T * pg).project_point(p).shape
            assert pg.project_point(p) == approx((T * pg).project_point(p))
            # Translating the detector by (v,u) should reduce project_point by (1, 1):
            T = ts.translate(pg.det_v + pg.det_u)
            assert pg.project_point(p) - (T * pg).project_point(p) == approx(
                np.ones((pg.num_angles, 2))
            )
