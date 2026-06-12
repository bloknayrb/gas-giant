"""DETAIL_FX variant: longitudinal intermittency gate (and, from the
hero-spiral commit, the GRS internal lanes).

Cross-variant comparisons use atol, never byte-equality (different
binaries may reschedule FP in shared expressions); byte-equality is only
asserted within one program."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.validate import validate_arrays

pytestmark = pytest.mark.gpu


def _quick_params(**detail) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.detail.intensity = 0.8
    p.detail.striation_amount = 0.6
    for key, value in detail.items():
        setattr(p.detail, key, value)
    return p


def test_explicit_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(intermittency=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_forced_fx_variant_is_noop_at_epsilon(gpu):
    """intermittency=1e-6 forces the DETAIL_FX program while the gate stays
    1 +- 1.5e-6 — the test that actually exercises the variant."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_quick_params(intermittency=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


def _synth_detail_field(gpu, params, size=(1024, 512)) -> np.ndarray:
    """Run DetailSynth directly (the composed color map mixes in the
    ungated cells and the T2-tracer term, which drown the gated filament
    signal in a ratio statistic)."""
    from gasgiant.engine.snapshot import hero_centers

    sim = Simulation(params, gpu)
    s = sim.solver
    out = gpu.texture2d(size, 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, params.detail,
        heroes=hero_centers(sim.vortices),
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field


def test_intermittency_adds_longitudinal_patchiness_not_tone(gpu):
    base = _synth_detail_field(gpu, _quick_params())
    gated = _synth_detail_field(gpu, _quick_params(intermittency=0.9))
    assert not np.array_equal(base, gated)
    assert np.all(np.isfinite(gated))
    assert abs(float(gated.mean() - base.mean())) < 0.02  # patchiness, not tone

    # Intermittent modulation: per-block RATIO of filament energy
    # gated/base (16x16 blocks over band latitudes). Without the gate every
    # ratio is exactly 1; the gate multiplies the filament weight by
    # ~[0.32, 1.6] in a low-frequency pattern, so the ratios must SPREAD
    # around ~1 (some blocks calmed, some boosted) — patchiness, directly,
    # independent of how much longitudinal structure the base already had.
    def block_energy(field: np.ndarray) -> np.ndarray:
        h = field.shape[0]
        band = field[int(0.25 * h): int(0.75 * h)] - 0.5
        bh, bw = band.shape[0] // 16, band.shape[1] // 16
        blocks = band[: bh * 16, : bw * 16].reshape(bh, 16, bw, 16)
        return np.sqrt((blocks ** 2).mean(axis=(1, 3)))

    e_base = block_energy(base)
    e_gated = block_energy(gated)
    active = e_base > np.quantile(e_base, 0.5)   # blocks with real texture
    ratio = e_gated[active] / e_base[active]
    assert 0.5 < float(ratio.mean()) < 1.5       # modulation, not a tone shift
    assert float(ratio.std()) > 0.1, ratio.std()  # ...and it is SPATIALLY patchy
    assert float(ratio.min()) < 0.7               # some calm runs
    assert float(ratio.max()) > 1.2               # some busy patches


def test_intermittency_render_is_seam_clean(gpu):
    sim = Simulation(_quick_params(intermittency=0.9), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems


# -- hero spiral -------------------------------------------------------------------


def test_hero_spiral_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(hero_spiral=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_hero_spiral_without_heroes_is_noop(gpu):
    """hero_spiral > 0 forces the FX variant, so this is a cross-variant
    comparison (atol, not byte-equality)."""
    p_base = _quick_params()
    p_base.storms.hero_count = 0
    p_spiral = _quick_params(hero_spiral=1.0)
    p_spiral.storms.hero_count = 0
    base = Simulation(p_base, gpu).render_maps(256)["color"]
    spiral = Simulation(p_spiral, gpu).render_maps(256)["color"]
    assert np.allclose(base, spiral, atol=1e-3), np.abs(base - spiral).max()


def test_hero_spiral_localized_to_hero_window(gpu):
    """Pinned configuration: intermittency 1e-6 in BOTH renders (same FX
    program, identical gate path), only hero_spiral varies. The q >= 1.9
    early-out adds exactly 0.0 outside the window, so far pixels are
    byte-equal; inside they differ."""
    base = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    spiral = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_spiral=1.2)
    )
    sim = Simulation(_quick_params(), gpu)
    heroes = sim.vortices.heroes()
    assert heroes, "seed 42 must seed a hero for this test"

    h, w = base.shape
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    cl = np.cos(lat)[:, None]
    px = cl * np.cos(lon)[None, :]
    py = np.sin(lat)[:, None] * np.ones((1, w))
    pz = cl * np.sin(lon)[None, :]
    inside = np.zeros((h, w), dtype=bool)
    for v in heroes:
        c = np.array([
            np.cos(v.lat) * np.cos(v.lon), np.sin(v.lat), np.cos(v.lat) * np.sin(v.lon)
        ])
        q = np.arccos(np.clip(px * c[0] + py * c[1] + pz * c[2], -1, 1)) / v.r_core
        inside |= q < 1.9

    np.testing.assert_array_equal(base[~inside], spiral[~inside])
    assert not np.array_equal(base[inside], spiral[inside])
    assert np.all(np.isfinite(spiral))


def test_hero_spiral_render_is_seam_clean(gpu):
    sim = Simulation(_quick_params(hero_spiral=1.0, intermittency=0.65), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems


def test_belt_texture_noop_at_epsilon_and_adds_belt_structure(gpu):
    base = _synth_detail_field(gpu, _quick_params())
    eps = _synth_detail_field(gpu, _quick_params(belt_texture=1e-6))
    assert np.allclose(base, eps, atol=1e-3)

    on = _synth_detail_field(gpu, _quick_params(belt_texture=1.0))
    assert np.all(np.isfinite(on))
    # The fold term + filament floor raise detail variance where belt_mask
    # is set; globally the variance must rise and the field must change.
    assert not np.array_equal(base, on)
    assert float(on.std()) > float(base.std()) * 1.05


def test_mottle_noop_at_epsilon_and_is_latitude_windowed(gpu):
    base = _synth_detail_field(gpu, _quick_params())
    eps = _synth_detail_field(gpu, _quick_params(mottle=1e-6))
    assert np.allclose(base, eps, atol=1e-3)

    on = _synth_detail_field(gpu, _quick_params(mottle=1.2))
    assert np.all(np.isfinite(on))
    h = on.shape[0]
    # 40-55 deg rows gain energy...
    band = slice(int((90 - 55) / 180 * h), int((90 - 40) / 180 * h))
    assert float(np.abs(on[band] - base[band]).mean()) > 1e-3
    # ...the equatorial tenth is untouched by the window (byte-equality is
    # valid because both renders force the SAME FX program).
    eq = slice(int(0.45 * h), int(0.55 * h))
    on_eq = _synth_detail_field(gpu, _quick_params(mottle=1.2, intermittency=1e-6))
    base_eq = _synth_detail_field(gpu, _quick_params(mottle=1e-6, intermittency=1e-6))
    np.testing.assert_array_equal(on_eq[eq], base_eq[eq])


def test_new_detail_knobs_are_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams()
    b = a.model_copy(deep=True)
    b.detail.belt_texture = 0.7
    b.detail.mottle = 0.5
    b.detail.belt_texture_fine = 0.9
    assert diff_tiers(a, b) == {Tier.POST}


def test_belt_texture_fine_noop_at_epsilon_and_adds_structure(gpu):
    # Force the FX variant in both arms via a tiny intermittency so we compare
    # the same compiled program (mirrors the mottle test pattern at lines 189-191).
    base = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    eps = _synth_detail_field(gpu, _quick_params(belt_texture_fine=1e-6,
                                                 intermittency=1e-6))
    assert np.allclose(base, eps, atol=1e-3)

    on = _synth_detail_field(gpu, _quick_params(belt_texture_fine=1.0,
                                                intermittency=1e-6))
    assert np.all(np.isfinite(on))
    assert not np.array_equal(base, on)
    assert float(on.std()) > float(base.std()) * 1.05


def test_belt_texture_fine_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(belt_texture_fine=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)
