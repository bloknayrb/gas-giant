"""T11 imported paint mask -> POST art-direction targets (GPU).

Byte-identity contract: with a mask BOUND but every gain 0 the MASK program is
NOT selected (mask_on keys on the gains), so the default program renders and the
output is byte-identical to no mask at all -- the forced-MASK no-op gate the CI
gpu-smoke job selects on "noop". A tiny gain forces the MASK program and stays a
no-op within the cross-variant tolerance (a different binary may reschedule FP,
the repo's documented reason variants are atol-compared, never byte-equal).

Per-target behavior tests then confirm each gain moves the output in the expected
direction, and a tiled-vs-whole test confirms the mask-derived tiles reassemble
to the whole-map mask render (the tile-apron contract, extended to the mask).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

_RES = 512


def _params(**mask_kw) -> PlanetParams:
    p = PlanetParams(seed=4)
    p.sim.resolution = _RES
    p.sim.dev_steps = 0
    for key, value in mask_kw.items():
        setattr(p.mask, key, value)
    return p


def _split_mask(w: int, h: int) -> np.ndarray:
    """A 2:1 mask: left half 0.0, right half 1.0 (a sharp longitude split)."""
    m = np.zeros((h, w), dtype=np.float32)
    m[:, w // 2:] = 1.0
    return m


def _uniform_mask(w: int, h: int, value: float) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


# -- forced-MASK no-op byte-identity (gpu-smoke selects these) -------------------


def test_mask_bound_zero_gains_is_noop_identical(gpu):
    """A mask bound with ALL gains 0 keeps mask_on False, so the DEFAULT program
    renders and the output is byte-identical to the no-mask render."""
    base = Simulation(_params(), gpu).render_maps(256)["color"]

    sim = Simulation(_params(), gpu)  # gains all 0 by default
    sim.set_mask(_split_mask(256, 128))
    masked = sim.render_maps(256)["color"]

    np.testing.assert_array_equal(base, masked)


def test_forced_mask_variant_is_noop_at_epsilon(gpu):
    """detail_gain=1e-6 forces the MASK program compiled/selected while leaving
    the math a no-op up to fp32 -- the test that actually exercises the variant
    (defaults route to the unchanged default text)."""
    base = Simulation(_params(), gpu).render_maps(256)["color"]

    sim = Simulation(_params(detail_gain=1e-6), gpu)
    sim.set_mask(_uniform_mask(256, 128, 0.5))
    fx = sim.render_maps(256)["color"]

    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


# -- per-target behavior --------------------------------------------------------


def test_detail_gain_darkens_masked_regions(gpu):
    """detail_gain multiplies luminance by mix(1, mask, gain): where the mask is
    0 the region darkens; where the mask is 1 it is untouched."""
    base = Simulation(_params(), gpu).render_maps(_RES)["color"][..., :3]

    sim = Simulation(_params(detail_gain=1.0), gpu)
    sim.set_mask(_split_mask(_RES, _RES // 2))
    out = sim.render_maps(_RES)["color"][..., :3]

    w = _RES
    left = slice(0, w // 4)          # mask ~0 -> factor ~0 -> darkened
    right = slice(3 * w // 4, w)     # mask ~1 -> factor 1 -> unchanged
    assert out[:, left].mean() < 0.25 * base[:, left].mean()
    assert np.allclose(out[:, right], base[:, right], atol=1e-3)


def test_band_fade_only_changes_painted_regions(gpu):
    """band_fade lerps toward the plain band color weighted by mask*gain: the
    mask-0 half is byte-unchanged (weight 0), the mask-1 half is altered."""
    base = Simulation(_params(), gpu).render_maps(_RES)["color"][..., :3]

    sim = Simulation(_params(band_fade=1.0), gpu)
    sim.set_mask(_split_mask(_RES, _RES // 2))
    out = sim.render_maps(_RES)["color"][..., :3]

    w = _RES
    left = slice(0, w // 2 - 2)       # mask 0 -> weight 0 -> identical
    right = slice(w // 2 + 2, w)      # mask 1 -> faded toward band color
    np.testing.assert_array_equal(out[:, left], base[:, left])
    assert not np.allclose(out[:, right], base[:, right], atol=1e-3)


def test_emission_gain_dims_masked_emission(gpu):
    """emission_gain modulates the emission map by mix(1, mask, gain): the mask-0
    region's night-side glow is dimmed toward 0; the mask-1 region is untouched."""
    p = _params(emission_gain=1.0)
    p.emission.thermal_strength = 1.0
    p.emission.lightning_strength = 0.5
    base_p = _params()
    base_p.emission.thermal_strength = 1.0
    base_p.emission.lightning_strength = 0.5

    base = Simulation(base_p, gpu).render_maps(_RES)["emission"]

    sim = Simulation(p, gpu)
    sim.set_mask(_split_mask(_RES, _RES // 2))
    out = sim.render_maps(_RES)["emission"]

    w = _RES
    left = slice(0, w // 4)          # mask 0 -> emission scaled to ~0
    right = slice(3 * w // 4, w)     # mask 1 -> unchanged
    assert out[:, left].sum() < 0.05 * max(base[:, left].sum(), 1e-6)
    assert np.allclose(out[:, right], base[:, right], atol=1e-4)


# -- tiled vs whole -------------------------------------------------------------


def test_mask_tiles_match_whole_render(gpu):
    """The mask-derived tiles must reassemble to the whole-map mask render (the
    mask is a pure function of global uv, so tiling can never disagree)."""
    from gasgiant.export.exporter import TILE, derive_tile

    p = _params(detail_gain=0.8, band_fade=0.3)
    p.export.width = 2048  # 2x1 tiles at TILE=1024
    sim = Simulation(p, gpu)
    sim.set_mask(_split_mask(1024, 512))
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


def test_mask_fields_are_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.mask.band_fade = 0.5
    b.mask.detail_gain = 0.3
    b.mask.file = "brush.png"
    assert diff_tiers(a, b) == {Tier.POST}


# -- painted-mask retention across a RESTART rebuild ----------------------------


def test_painted_mask_survives_restart_rebuild(gpu):
    """A GUI-painted mask has NO params.mask.file backing (strokes are app
    sidecar data): a RESTART-tier edit rebuilds every GL resource, and the
    facade must restore the explicitly-set mask from its retained CPU copy
    instead of silently wiping it via the file-driven sync."""
    sim = Simulation(_params(band_fade=0.8), gpu)
    unmasked = sim.render_maps(256)["color"]
    sim.set_mask(_uniform_mask(256, 128, 1.0))
    masked = sim.render_maps(256)["color"]
    assert not np.array_equal(unmasked, masked)  # premise: the mask acts

    p = sim.params.model_copy(deep=True)
    p.seed = 5  # RESTART tier: full rebuild
    sim.update_params(p)
    assert sim._mask_tex is not None  # restored, not wiped
    after = sim.render_maps(256)["color"]

    control = Simulation(sim.params.model_copy(deep=True), gpu)
    control_none = control.render_maps(256)["color"]
    assert not np.array_equal(after, control_none)  # mask still applied

    sim.set_mask(None)  # explicit clear FORGETS the painted mask
    p2 = sim.params.model_copy(deep=True)
    p2.seed = 6
    sim.update_params(p2)
    assert sim._mask_tex is None
