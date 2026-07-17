"""DETAIL_CHROMA derive variant: appearance.detail_chroma, the L-preserving
two-material tint keyed to the synthesized detail field's signed excursion
(bright detail -> designed cool, dark -> 0.3x warm; derive.comp, own axis).

Cross-variant comparisons use atol, never byte-equality (house rule — a
different binary may reschedule FP in shared expressions). detail_chroma=0
selects the PRIOR program, so byte-equality is legal exactly there.

The (CHROMA_FX off, DETAIL_CHROMA on) combination is deliberately the shape of
most tests here: it is the include-trap regression (oklab.glsl arrives via ONE
compound-guard include; a second guarded include would leave this exact
variant without the Oklab functions and it would fail to compile).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.palette.gradient import srgb_to_oklab
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _quick_params(**overrides) -> PlanetParams:
    """seed-4 dev-0 kinematic quick scene (test_chroma_fx.py harness); the
    default detail.intensity 0.55 > 0 means detail synthesis runs."""
    p = PlanetParams(seed=4)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    for key, value in overrides.items():
        setattr(p.appearance, key, value)
    return p


def test_zero_routes_to_default_program_byte_equal(gpu):
    """detail_chroma=0 selects the pre-existing program (the define is never
    set) — byte-equal, same-process."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    off = Simulation(_quick_params(detail_chroma=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, off)


def test_forced_variant_is_noop_at_epsilon(gpu):
    """detail_chroma=1e-6 forces the DETAIL_CHROMA program while leaving the
    push sub-fp32 — the test that actually exercises the variant text, and
    (chroma params all neutral) the compile regression for the
    CHROMA_FX-off/DETAIL_CHROMA-on include-trap combination."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    eps = Simulation(_quick_params(detail_chroma=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, eps, atol=2e-3), np.abs(base - eps).max()


def test_pushes_bright_detail_cool_and_dark_detail_weakly_warm(gpu):
    """The designed asymmetric material push, measured directly in Oklab b
    (blue-yellow): pixels the detail field LIFTS must move toward blue
    (delta-b < 0), pixels it darkens toward yellow/warm, weaker (0.3x). The
    detail excursion sign per pixel is recovered by differencing renders at
    detail.intensity on/off — the same scalar the shader reads. A
    chroma-MAGNITUDE metric would pass a monochrome-warm failure; signed b
    per excursion group is the review-mandated hue test."""
    p_nodet = _quick_params()
    p_nodet.detail.intensity = 0.0
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    no_det = Simulation(p_nodet, gpu).render_maps(256)["color"][..., :3] @ lum_w
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"][..., :3]
    on = Simulation(_quick_params(detail_chroma=1.0), gpu).render_maps(256)["color"][..., :3]

    ex_proxy = (base @ lum_w) - no_det  # sign of the detail excursion
    lab_base = srgb_to_oklab(np.clip(base, 0, 1).reshape(-1, 3)).reshape(*base.shape[:2], 3)
    lab_on = srgb_to_oklab(np.clip(on, 0, 1).reshape(-1, 3)).reshape(*on.shape[:2], 3)
    db = lab_on[..., 2] - lab_base[..., 2]

    bright = ex_proxy > np.quantile(ex_proxy, 0.9)
    dark = ex_proxy < np.quantile(ex_proxy, 0.1)
    assert db[bright].mean() < -1e-3, db[bright].mean()   # cool push
    assert db[dark].mean() > 2e-4, db[dark].mean()        # weaker warm push
    assert abs(db[bright].mean()) > 2.0 * abs(db[dark].mean())  # asymmetry
    # L-preserving: whole-image luma barely moves (gamut clamp only).
    assert abs(float((on @ lum_w).mean() - (base @ lum_w).mean())) < 5e-3


def test_composes_with_chroma_fx(gpu):
    """Both axes on = a distinct 6-tuple program; must compile and stay
    finite (oklab.glsl shared through the single hoisted include)."""
    both = Simulation(
        _quick_params(detail_chroma=0.8, chroma_scale=1.3, chroma_variance=0.3), gpu
    ).render_maps(256)["color"]
    assert np.all(np.isfinite(both))


def test_noop_without_detail_intensity(gpu):
    """detail.intensity=0 gates the whole dsyn block (this is also what makes
    the cube projection neutral — the exporter passes intensity 0 for cube):
    the lever must do nothing without a detail field."""
    p_off = _quick_params()
    p_off.detail.intensity = 0.0
    p_on = _quick_params(detail_chroma=1.0)
    p_on.detail.intensity = 0.0
    a = Simulation(p_off, gpu).render_maps(256)["color"]
    b = Simulation(p_on, gpu).render_maps(256)["color"]
    assert np.allclose(a, b, atol=2e-3), np.abs(a - b).max()


def test_detail_chroma_is_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.appearance.detail_chroma = 0.5
    assert diff_tiers(a, b) == {Tier.POST}
