#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for phantom generation."""


from pytest import approx
import tomosipo as ts


def test_hollow_box():
    """Test something."""

    for s in [10, 20, 30]:
        vd = ts.data(ts.volume(shape=100))
        ts.phantom.hollow_box(vd)
        assert vd.data.mean() == approx(0.208)
