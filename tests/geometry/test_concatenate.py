"""Tests for ConeGeometry."""

import pytest
from pytest import approx
import numpy as np
import tomosipo as ts
import tomosipo.geometry as g
import tomosipo.vector_calc as vc
from tomosipo.geometry import transform


@pytest.mark.parametrize("t", [g.random_transform(), g.random_box()])
def test_transform(t):
    assert ts.concatenate([t, t]).num_steps == 2 * t.num_steps
    tt = ts.concatenate([t, t])
    assert tt[: t.num_steps] == t


random_geometries = [
    g.random_cone(),
    g.random_cone_vec(),
    g.random_parallel(),
    g.random_parallel_vec(),
]


@pytest.mark.parametrize("t", random_geometries)
def test_geometries(t):
    assert ts.concatenate([t, t]).num_angles == 2 * t.num_angles
    tt = ts.concatenate([t, t])
    assert tt[: t.num_angles] == t.to_vec()


def test_concat_errors():
    with pytest.raises(TypeError):
        ts.concatenate([ts.volume()])
        ts.concatenate(ts.volume())
        ts.concatenate(ts.cone())
    with pytest.raises(ValueError):
        ts.concatenate([])
