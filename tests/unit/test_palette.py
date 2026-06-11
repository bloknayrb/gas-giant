from __future__ import annotations

import numpy as np
import pytest

from gasgiant.palette import bake_lut


def test_endpoints_clamp():
    lut = bake_lut([(0.2, (1.0, 0.0, 0.0)), (0.8, (0.0, 0.0, 1.0))], size=256)
    assert np.allclose(lut[0, :3], (1.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(lut[-1, :3], (0.0, 0.0, 1.0), atol=1e-6)


def test_midpoint_interpolation():
    lut = bake_lut([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))], size=256)
    mid = lut[128, :3]
    assert np.allclose(mid, 0.5, atol=0.01)


def test_unsorted_stops_accepted():
    lut = bake_lut([(1.0, (1.0, 1.0, 1.0)), (0.0, (0.0, 0.0, 0.0))], size=64)
    assert lut[0, 0] < lut[-1, 0]


def test_monotonic_for_monotonic_stops():
    lut = bake_lut([(0.0, (0.0, 0.0, 0.0)), (0.5, (0.4, 0.4, 0.4)), (1.0, (1.0, 1.0, 1.0))])
    assert np.all(np.diff(lut[:, 0]) >= -1e-7)


def test_alpha_is_one():
    lut = bake_lut([(0.5, (0.3, 0.3, 0.3))])
    assert np.all(lut[:, 3] == 1.0)


def test_empty_stops_rejected():
    with pytest.raises(ValueError):
        bake_lut([])
