#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for utils."""


import pytest
from pytest import approx
from tomosipo.utils import up_tuple, to_shape, to_pos, to_size, slice_interval
import numpy as np
import tomosipo as ts


class Interval(object):
    """Documentation for Interval

    """

    def __init__(self, l, r, length):
        super(Interval, self).__init__()
        self.l = l
        self.r = r
        # self.l = np.array(l, ndmin=1)
        # self.r = np.array(r, ndmin=1)
        self.length = length

    def __getitem__(self, key):
        l, r, length, _ = slice_interval(self.l, self.r, self.length, key)
        return Interval(l, r, length)

    def __eq__(self, other):
        return (
            self.length == other.length
            and np.all(abs(self.l - other.l) < ts.epsilon)
            and np.all(abs(self.r - other.r) < ts.epsilon)
        )

    def __repr__(self):
        return f"Interval({self.l}, {self.r}, {self.length}))"

    @property
    def size(self):
        return self.r - self.l

    @property
    def pixel_size(self):
        return self.size / self.length

    @property
    def center_of_left_pixel(self):
        return self.l + self.pixel_size / 2

    @property
    def center(self):
        return (self.l + self.r) / 2


def test_up_tuple():
    """Test up_tuple."""

    assert (1, 1) == up_tuple((1, 1), 2)
    assert (1, 2) == up_tuple((1, 2), 2)
    assert (1, 1) == up_tuple([1], 2)
    assert (0, 0) == up_tuple(range(1), 2)
    assert (0, 0) == up_tuple([0], 2)
    assert (0, 0) == up_tuple((0,), 2)
    assert (0, 0) == up_tuple(iter((0,)), 2)


def test_to_shape():
    assert (1, 1, 1) == to_shape(1)
    assert (0, 0, 0) == to_shape(0)
    assert (1, 1) == to_shape(1, dim=2)
    with pytest.raises(ValueError):
        to_shape((1,2,3,4))
    with pytest.raises(TypeError):
        to_shape(1.0)
    with pytest.raises(ValueError):
        to_shape(-1)


def test_to_pos():
    assert (0, 0, 0) == to_pos(0)
    assert (0, 0) == to_pos(0, dim=2)
    assert (1, 2, 3) == to_pos((1, 2, 3))
    two_rows = np.random.normal(size=(2, 3))
    assert np.allclose(two_rows, to_pos(two_rows))
    with pytest.raises(ValueError):
        to_pos(1.0)


def test_to_size():
    assert (0, 0, 0) == to_size(0)
    assert (0, 0) == to_size(0, dim=2)
    assert (1, 2, 3) == to_size((1, 2, 3))

    two_rows = abs(np.random.normal(size=(2, 3)))
    assert np.allclose(two_rows, to_size(two_rows))

    with pytest.raises(ValueError):
        to_size((-1, 0, 0))


def test_slice_interval():
    (l, r, length, ps) = slice_interval(0, 1, 1, 0)
    assert (l, r, length, ps) == (0, 1, 1, 1)

    (l, r, length, ps) = slice_interval(0, 1, 1, slice(None, None, None))
    assert (l, r, length, ps) == (0, 1, 1, 1)

    (l, r, length, ps) = slice_interval(0, 0, 1, 0)
    assert (l, r, length, ps) == (0, 0, 1, 0)

    (l, r, length, ps) = slice_interval(0, 0, 0, 0)
    assert (l, r, length, ps) == (0, 0, 0, 1)

    # Test with l and r as np.arrays
    l0 = np.arange(10)
    r0 = np.arange(10) + 1
    length0 = 10
    key = slice(None, None, None)

    l1, r1, length1, ps1 = slice_interval(l0, r0, length0, key)

    assert approx(0.0) == np.sum(abs(l0 - l1))
    assert approx(0.0) == np.sum(abs(r0 - r1))
    assert length0 == length1

    # Check that moving the start with a step size translates the
    # interval:
    st = Interval(0, 10, 10)
    for i in range(1, 5):
        assert st[::5] != st[i::5]
        assert st[::5].length == st[i::5].length
        assert approx(st[::5].l + i) == st[i::5].l
        assert approx(st[::5].r + i) == st[i::5].r
        assert approx(st[::5].center + i) == st[i::5].center
        assert st[::5].size == st[i::5].size

    # Check what happens to the left-most pixel as we take a larger step size:
    for step in range(1, 10):
        assert st[::step].center_of_left_pixel == approx(st.center_of_left_pixel)

    # Check that length is consistent with np indexing:
    for length in [0, 5, 6]:
        st = Interval(0, 1, length)
        ones = np.ones(length)
        for start in range(length):
            for step in [1, 2, 3, 4]:
                assert st[start::step].length == len(ones[start::step])
                assert st[start::step].pixel_size == approx(step * st.pixel_size)

    st = Interval(0, 1, 30)
    assert st[::2].pixel_size == approx(2 * st.pixel_size)
    assert st[1::2].size == approx(1.0)
    assert st[2::3].size == approx(1.0)
    assert st[4::5].size == approx(1.0)

    # Check that the center is preserved with step size 3:
    assert st[1::3].center == approx(st.center)
    assert st[0::3].center != approx(st.center)
    assert st[2::3].center != approx(st.center)
