"""T14 per-band RGB tint -> final art-direction override (GPU).

Byte-identity contract: with band_tint_strength 0 the BAND_TINT program is NOT
selected (band_tint_on keys on strength), so the DEFAULT program renders and the
output is byte-identical to no tint at all -- the no-op gate the CI gpu-smoke job
selects on "noop". A tiny strength forces the BAND_TINT program and stays a no-op
within the cross-variant tolerance (a recompiled binary may reschedule FP, so
variants are atol-compared, never byte-equal).

A latitude-row test then confirms a single stop recolors only its band, and a
tiled-vs-whole test confirms the tint (a pure function of global uv) reassembles.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import GradientStop, PlanetParams

pytestmark = pytest.mark.gpu

_RES = 512


def _params(strength: float = 0.0, stops=None) -> PlanetParams:
    p = PlanetParams(seed=4)
    p.sim.resolution = _RES
    p.sim.dev_steps = 0
    p.appearance.band_tint_strength = strength
    if stops is not None:
        p.appearance.band_tint_stops = stops
    return p


# -- no-op byte-identity (gpu-smoke selects "noop") ------------------------------


def test_band_tint_strength_zero_is_noop_identical(gpu):
    """band_tint_strength 0 keeps band_tint_on False, so the DEFAULT program
    renders and the output is byte-identical to no tint -- even with non-neutral
    stops loaded (the predicate keys on strength, not the stops)."""
    base = Simulation(_params(), gpu).render_maps(256)["color"]

    loaded = _params(
        strength=0.0,
        stops=[
            GradientStop(pos=0.0, color=(1.0, 0.0, 0.0)),
            GradientStop(pos=1.0, color=(0.0, 0.0, 1.0)),
        ],
    )
    tinted = Simulation(loaded, gpu).render_maps(256)["color"]

    np.testing.assert_array_equal(base, tinted)


def test_forced_band_tint_variant_is_noop_at_epsilon(gpu):
    """strength=1e-6 forces the BAND_TINT program compiled/selected while leaving
    the mix a no-op up to fp32 (neutral gray stops) -- the test that actually
    exercises the variant (defaults route to the unchanged default text)."""
    base = Simulation(_params(), gpu).render_maps(256)["color"]

    fx = Simulation(_params(strength=1e-6), gpu).render_maps(256)["color"]

    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


# -- latitude-row behavior ------------------------------------------------------


def test_band_tint_stop_recolors_only_its_band(gpu):
    """The tint LUT is indexed by 1 - uv.y (north at the top). Against an
    all-gray tint (whole planet flat gray at strength 1), reddening only the
    NORTHERN stops recolors the top rows while the southern half stays byte-equal
    -- a per-latitude override, not a global recolor."""
    gray = [
        GradientStop(pos=0.0, color=(0.5, 0.5, 0.5)),
        GradientStop(pos=0.5, color=(0.5, 0.5, 0.5)),
        GradientStop(pos=1.0, color=(0.5, 0.5, 0.5)),
    ]
    # South half (pos 0..0.5) stays gray; north half (0.5..1.0) ramps to red.
    north_red = [
        GradientStop(pos=0.0, color=(0.5, 0.5, 0.5)),
        GradientStop(pos=0.5, color=(0.5, 0.5, 0.5)),
        GradientStop(pos=1.0, color=(0.9, 0.1, 0.1)),
    ]
    base = Simulation(_params(strength=1.0, stops=gray), gpu).render_maps(_RES)["color"][..., :3]
    out = Simulation(
        _params(strength=1.0, stops=north_red), gpu
    ).render_maps(_RES)["color"][..., :3]

    h = _RES // 2
    top = slice(0, h // 4)              # north -> reddened
    bottom = slice(3 * h // 4, h)       # south -> gray, byte-unchanged
    np.testing.assert_array_equal(out[bottom], base[bottom])
    # North rows: redder (r up, g/b down) than the flat-gray baseline.
    assert out[top, :, 0].mean() > base[top, :, 0].mean() + 0.1
    assert out[top, :, 2].mean() < base[top, :, 2].mean() - 0.1


# -- tiled vs whole -------------------------------------------------------------


def test_band_tint_tiles_match_whole_render_identical(gpu):
    """The tint is a pure function of global uv (1 - uv.y), so the band-tinted
    tiles must reassemble byte-identically to the whole-map tinted render."""
    from gasgiant.export.exporter import TILE, derive_tile

    stops = [
        GradientStop(pos=0.0, color=(0.2, 0.3, 0.6)),
        GradientStop(pos=1.0, color=(0.8, 0.6, 0.2)),
    ]
    p = _params(strength=0.5, stops=stops)
    p.export.width = 2048  # 2x1 tiles at TILE=1024
    sim = Simulation(p, gpu)
    sim.run_to_completion()

    w, h = 2048, 1024
    whole = sim.render_maps(w)["color"]

    snap = sim.create_snapshot()
    tc = gpu.texture2d((TILE, TILE), 4, "f4")
    th = gpu.texture2d((TILE, TILE), 1, "f4")
    td = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
    tiled = np.empty((h, w, 4), dtype=np.float32)
    try:
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                tw = min(TILE, w - x0)
                thh = min(TILE, h - y0)
                derive_tile(sim, snap, snap.params, x0, y0, w, h, tc, th, td, None)
                tiled[y0:y0 + thh, x0:x0 + tw] = gpu.read_texture(tc)[:thh, :tw]
    finally:
        tc.release()
        th.release()
        td.release()
        snap.release()

    np.testing.assert_array_equal(tiled, whole)


# -- tier ------------------------------------------------------------------------


def test_band_tint_fields_are_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.appearance.band_tint_strength = 0.5
    b.appearance.band_tint_stops = [
        GradientStop(pos=0.0, color=(0.2, 0.3, 0.4)),
        GradientStop(pos=1.0, color=(0.9, 0.7, 0.5)),
    ]
    assert diff_tiers(a, b) == {Tier.POST}
