"""gasgiant.palette.agx: numpy port of the repo's AgX approximation.

The gray-neutrality test is the load-bearing one: GLSL mat3 constructors
are column-major, so the single most likely porting bug is a transposed
matrix — which passes monotonicity/range tests but breaks neutrality
(gray in, visibly chromatic out).
"""

from __future__ import annotations

import numpy as np

from gasgiant.palette.agx import agx_constants_checksum, agx_view, quartile_chroma_retention


def test_gray_axis_neutral():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    out = agx_view(np.stack([ramp, ramp, ramp], axis=-1))
    assert np.all(out.max(axis=-1) - out.min(axis=-1) < 1e-3)


def test_gray_axis_monotonic():
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    out = agx_view(np.stack([ramp, ramp, ramp], axis=-1))
    lum = out.mean(axis=-1)
    assert np.all(np.diff(lum) >= -1e-6)


def test_black_and_white_endpoints():
    out = agx_view(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32))
    assert np.all(out[0] < 0.02)          # black stays black
    assert np.all((out[1] > 0.74) & (out[1] < 0.83))  # AgX white shoulder ~0.787


def test_out_of_range_inputs_finite_and_clipped():
    bad = np.array([[-0.2, 1.4, 0.5], [2.0, -1.0, 0.0]], dtype=np.float32)
    out = agx_view(bad)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_vectorized_matches_per_pixel():
    rng = np.random.default_rng(3)
    img = rng.random((8, 8, 3)).astype(np.float32)
    full = agx_view(img)
    loop = np.stack([
        np.stack([agx_view(img[y, x]) for x in range(img.shape[1])])
        for y in range(img.shape[0])
    ])
    assert np.allclose(full, loop, atol=1e-6)


def test_checksum_stable_and_short():
    c = agx_constants_checksum()
    assert c == agx_constants_checksum()
    assert len(c) == 12


def test_quartile_chroma_retention_known_directions():
    """v1.2 review measurement: AgX barely desaturates dark belt tones but
    halves pale-zone chroma. The retention helper must reproduce that
    direction on a synthetic belt/zone image."""
    # Left half belt / right half zone so EVERY latitude bin contains both
    # populations (horizontal bands would average all-zone bins into the
    # belt retention figure).
    img = np.full((60, 120, 3), (0.85, 0.80, 0.72), dtype=np.float32)  # pale zone
    img[:, :60] = (0.45, 0.30, 0.18)  # dark rusty belt columns
    r = quartile_chroma_retention(img, bins=3, lat_max=90.0)
    assert r["belt"] > 0.8          # dark belt chroma survives AgX
    assert r["zone"] < 0.7          # pale zone chroma is AgX-limited
    assert r["belt"] > r["zone"]
