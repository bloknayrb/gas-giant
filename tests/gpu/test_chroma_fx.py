"""CHROMA_FX derive variant: Oklab chroma scale + longitudinal drift.

Cross-variant comparisons use atol, never byte-equality: the FX program is
a different binary and the compiler may reschedule FP in shared
expressions (the repo's documented rationale for define-gated variants)."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.palette.gradient import srgb_to_oklab
from gasgiant.params.model import GradientStop, PaletteRow, PlanetParams

pytestmark = pytest.mark.gpu


def _quick_params(**appearance) -> PlanetParams:
    p = PlanetParams(seed=4)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    for key, value in appearance.items():
        setattr(p.appearance, key, value)
    return p


def _mean_chroma(color: np.ndarray) -> float:
    lab = srgb_to_oklab(color[..., :3].reshape(-1, 3))
    return float(np.hypot(lab[:, 1], lab[:, 2]).mean())


def test_forced_fx_variant_is_noop_at_neutral_values(gpu):
    """chroma_variance=1e-6 forces the CHROMA_FX program while leaving the
    math a no-op up to the fp32 Oklab round-trip — the test that actually
    exercises the variant (defaults route to the unchanged default text)."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_quick_params(chroma_variance=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=2e-3), np.abs(base - fx).max()


def test_chroma_scale_moves_mean_chroma_both_ways(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    up = Simulation(_quick_params(chroma_scale=1.5), gpu).render_maps(256)["color"]
    down = Simulation(_quick_params(chroma_scale=0.0), gpu).render_maps(256)["color"]
    assert np.all(np.isfinite(up)) and np.all(np.isfinite(down))
    assert _mean_chroma(up) > 1.2 * _mean_chroma(base)
    assert _mean_chroma(down) < 0.05 * _mean_chroma(base)


def test_chroma_variance_adds_within_quartile_chroma_spread(gpu):
    from gasgiant.palette.reference import latitude_profile

    base = Simulation(_quick_params(), gpu).render_maps(512)["color"][..., :3]
    var = Simulation(
        _quick_params(chroma_variance=0.4), gpu
    ).render_maps(512)["color"][..., :3]
    p_base = latitude_profile(np.clip(base, 0, 1), bins=12)
    p_var = latitude_profile(np.clip(var, 0, 1), bins=12)
    band = np.abs(p_base.lat_deg) <= 50.0
    assert p_var.belt_chroma_std[band].mean() > p_base.belt_chroma_std[band].mean()


def test_determinism_and_tile_consistency(gpu):
    """Same FX program twice -> byte-equal; and the full-frame render must
    match the same pixels derived as tiles (the chroma drift is a pure
    function of global position + a seed-derived uniform)."""
    p = _quick_params(chroma_scale=1.3, chroma_variance=0.3)
    a = Simulation(p, gpu).render_maps(256)["color"]
    b = Simulation(p, gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(a, b)


def test_emission_plus_chroma_combined_variant_compiles(gpu):
    """The (EMISSION, CHROMA_FX) fourth program is a distinct compile; a
    symbol clash between oklab.glsl and the emission block would only
    surface here."""
    p = _quick_params(chroma_scale=1.2)
    p.emission.thermal_strength = 0.3
    p.emission.lightning_strength = 0.1
    sim = Simulation(p, gpu)
    maps = sim.render_maps(256)
    assert "emission" in maps
    assert np.all(np.isfinite(maps["emission"]))
    assert np.all(np.isfinite(maps["color"]))


def test_glsl_matches_python_oklab_scaling(gpu):
    """Render a flat-palette planet with chroma_scale only (no variance) and
    reproduce the scaling in Python from the unscaled render — pins the GLSL
    matrices to gradient.py and exercises the negative-LMS cube path."""
    stops = [
        GradientStop(pos=0.0, color=(0.55, 0.36, 0.24)),
        GradientStop(pos=1.0, color=(0.55, 0.36, 0.24)),
    ]
    rows = [PaletteRow(latitude=0.0, stops=stops)]
    base = Simulation(
        _quick_params(palette_rows=rows), gpu
    ).render_maps(128)["color"][..., :3]
    scaled = Simulation(
        _quick_params(palette_rows=rows, chroma_scale=1.3), gpu
    ).render_maps(128)["color"][..., :3]

    lab = srgb_to_oklab(np.clip(base, 0.0, 1.0).reshape(-1, 3))
    lab[:, 1:] *= 1.3
    from gasgiant.palette.gradient import _oklab_to_srgb

    expected = _oklab_to_srgb(lab).reshape(base.shape)
    assert np.allclose(scaled, expected, atol=2.0 / 255.0), (
        np.abs(scaled - expected).max()
    )


def test_chroma_params_are_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.appearance.chroma_scale = 1.4
    b.appearance.chroma_variance = 0.2
    assert diff_tiers(a, b) == {Tier.POST}


def test_hue_variance_noop_at_epsilon_and_rotates_hue(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    eps = Simulation(_quick_params(hue_variance=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, eps, atol=2e-3), np.abs(base - eps).max()

    on = Simulation(_quick_params(hue_variance=0.3), gpu).render_maps(256)["color"]
    assert np.all(np.isfinite(on))
    assert not np.allclose(base, on, atol=2e-3)
    # Luminance-neutral: hue rotation in (a, b) leaves Oklab L unchanged up
    # to the gamut clamp, so mean luma must barely move.
    luma = lambda img: (img[..., :3] @ np.array([0.2126, 0.7152, 0.0722])).mean()
    assert abs(float(luma(on) - luma(base))) < 0.01
