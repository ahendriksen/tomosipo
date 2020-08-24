#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for operator."""

import pytest
import numpy as np
import tomosipo as ts


def test_link_numpy():
    vg = ts.volume(shape=10)

    with pytest.raises(ValueError):
        x = np.zeros((9, 9, 9), dtype=np.float32)
        ts.link(vg, x)
