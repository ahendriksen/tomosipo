#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for astra conversions."""

import pytest
import tomosipo as ts
from tomosipo.geometry import (
    random_cone,
    random_cone_vec,
    random_parallel,
    random_parallel_vec,
    random_volume,
)


def test_to_from_astra():
    rs = [
        random_cone,
        random_cone_vec,
        random_parallel,
        random_parallel_vec,
        random_volume,
    ]

    for r in rs:
        g = r()
        g_astra = ts.to_astra(g)
        g_ts = ts.from_astra(g_astra)
        assert g == g_ts
        with pytest.raises(TypeError):
            ts.from_astra(g)
        with pytest.raises(TypeError):
            ts.to_astra(g_astra)
