from __future__ import annotations

import numpy as np
import pytest

from gasgiant.palette import bake_lut, bake_rows


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


# -- bake_rows -------------------------------------------------------------------

_WARM = [(0.0, (0.36, 0.27, 0.21)), (1.0, (0.93, 0.89, 0.81))]
_COOL = [(0.0, (0.20, 0.24, 0.30)), (1.0, (0.75, 0.80, 0.85))]


def test_single_row_is_bitwise_bake_lut_everywhere():
    """The format-2 promise: one row == the v1 flat palette, exactly."""
    rows_lut = bake_rows([(0.0, _WARM)], size=256, height=64)
    flat = bake_lut(_WARM, size=256)
    for i in range(64):
        assert np.array_equal(rows_lut[i], flat)


def test_rows_clamp_beyond_outermost_anchors():
    lut = bake_rows([(-30.0, _WARM), (30.0, _COOL)], size=64, height=64)
    warm = bake_lut(_WARM, size=64)
    cool = bake_lut(_COOL, size=64)
    assert np.array_equal(lut[0], warm)    # -90: clamps to the -30 anchor
    assert np.array_equal(lut[-1], cool)   # +90: clamps to the +30 anchor


def test_rows_blend_monotonically_between_anchors():
    lut = bake_rows([(-60.0, _WARM), (60.0, _COOL)], size=8, height=96)
    # Red falls and blue rises moving north through the blend region.
    red = lut[:, 4, 0]
    blue = lut[:, 4, 2]
    assert np.all(np.diff(red) <= 1e-6)
    assert np.all(np.diff(blue) >= -1e-6)
    assert red[0] > red[-1]
    assert blue[0] < blue[-1]


def test_rows_blend_midpoint_not_desaturated():
    # Oklab blending: halfway between saturated complements must keep
    # substantially more chroma than a naive sRGB lerp would.
    orange = [(0.0, (0.8, 0.45, 0.15)), (1.0, (0.8, 0.45, 0.15))]
    blue = [(0.0, (0.2, 0.4, 0.7)), (1.0, (0.2, 0.4, 0.7))]
    lut = bake_rows([(-50.0, orange), (50.0, blue)], size=4, height=64)
    mid = lut[32, 0, :3]
    assert mid.max() - mid.min() > 0.02  # not collapsed to gray
    assert 0.1 < mid.mean() < 0.9


def test_rows_alpha_and_shape():
    lut = bake_rows([(0.0, _WARM)], size=128, height=32)
    assert lut.shape == (32, 128, 4)
    assert lut.dtype == np.float32
    assert np.all(lut[..., 3] == 1.0)


def test_empty_rows_rejected():
    with pytest.raises(ValueError):
        bake_rows([])
