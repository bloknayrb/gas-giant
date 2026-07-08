"""Cirrus-fiber synthesis (DETAIL_FX): render-time combed-fiber strands over
the elongated bright-cloud stamps, plus the streak_mute accumulator kill.

Cross-variant comparisons use atol, never byte-equality (different binaries
may reschedule FP in shared expressions); byte-equality is only asserted
within one program (the test_detail_fx.py doctrine).
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.validate import validate_arrays

pytestmark = pytest.mark.gpu

# The fiber attenuation fades strands whose across-flow spacing nears the
# output pixel (spacing ~ r_core/freq radians), so the direct-synth field must
# be rendered LARGE enough that default-frequency strands are resolvable.
_SYNTH_SIZE = (4096, 2048)


def _cloud_params(**detail) -> PlanetParams:
    """Neptune-shaped storm population: elongated bright companions + accent
    (the bright_cloud_centers class), on the fast dev_steps=0 test config."""
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.detail.intensity = 0.8
    p.storms.hero_companions = 2
    p.storms.companion_aspect = 3.5
    p.storms.accent_count = 1
    p.storms.accent_aspect = 4.0
    p.storms.accent_brightness = 0.32
    p.storms.accent_radius = 0.06
    for key, value in detail.items():
        setattr(p.detail, key, value)
    return p


def _synth_detail_field(gpu, params, size=_SYNTH_SIZE) -> np.ndarray:
    """Run DetailSynth directly (the composed color map mixes in the ungated
    cells and the T2-tracer term, which drown the gated fiber signal), passing
    the cloud list + profile_stamp the fiber mask needs."""
    from gasgiant.engine.snapshot import bright_cloud_centers, hero_centers

    sim = Simulation(params, gpu)
    s = sim.solver
    out = gpu.texture2d(size, 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, params.detail,
        heroes=hero_centers(sim.vortices),
        clouds=bright_cloud_centers(sim.vortices),
        profile_stamp=sim.profile_stamp,
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field


def _cloud_windows(sim, shape) -> np.ndarray:
    """Boolean mask of the analytic fiber windows (q < 2.2 in the dilated
    east-west elliptical metric), mirroring the shader's early-out."""
    from gasgiant.engine.snapshot import bright_cloud_centers

    h, w = shape
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    cl = np.cos(lat)[:, None]
    px = cl * np.cos(lon)[None, :]
    py = np.sin(lat)[:, None] * np.ones((1, w))
    pz = cl * np.sin(lon)[None, :]
    inside = np.zeros((h, w), dtype=bool)
    for (cx, cy, cz, rc, asp) in bright_cloud_centers(sim.vortices):
        c = np.array([cx, cy, cz])
        ew = np.cross([0.0, 1.0, 0.0], c)
        e1 = ew / np.linalg.norm(ew)
        e2 = np.cross(c, e1)
        d1 = (px * e1[0] + py * e1[1] + pz * e1[2]) / (1.5 * max(asp, 1.0))
        d2 = px * e2[0] + py * e2[1] + pz * e2[2]
        q = np.hypot(d1, d2) / max(rc, 1e-4)
        near = (px * c[0] + py * c[1] + pz * c[2]) > 0.0
        inside |= (q < 2.2) & near
    return inside


def test_cirrus_zero_routes_to_default_program(gpu):
    base = Simulation(_cloud_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_cloud_params(cirrus_fibers=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_cirrus_forced_variant_is_noop_at_epsilon(gpu):
    """cirrus_fibers=1e-6 forces the DETAIL_FX program while the fiber
    amplitude stays ~1e-6 — the test that actually exercises the variant."""
    base = Simulation(_cloud_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_cloud_params(cirrus_fibers=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


def test_streak_mute_epsilon_noop_and_live(gpu):
    base = Simulation(_cloud_params(), gpu).render_maps(256)["color"]
    eps = Simulation(_cloud_params(streak_mute=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, eps, atol=1e-3), np.abs(base - eps).max()

    # mute=1.0 kills the whole filament-streak accumulator: with
    # detail.intensity 0.8 the synthesized field (and thus the color) moves.
    muted = _synth_detail_field(gpu, _cloud_params(streak_mute=1.0), size=(1024, 512))
    plain = _synth_detail_field(gpu, _cloud_params(), size=(1024, 512))
    assert not np.array_equal(plain, muted)
    assert np.all(np.isfinite(muted))
    # The muted field has strictly less streak energy, never more.
    assert float(muted.std()) < float(plain.std())


def test_cirrus_without_elongated_clouds_is_noop(gpu):
    """cirrus_fibers > 0 forces the FX variant, so this is a cross-variant
    comparison (atol). No companions/accents -> empty cloud list -> no-op."""
    p_base = _cloud_params()
    p_base.storms.hero_companions = 0
    p_base.storms.accent_count = 0
    p_on = _cloud_params(cirrus_fibers=1.0)
    p_on.storms.hero_companions = 0
    p_on.storms.accent_count = 0
    base = Simulation(p_base, gpu).render_maps(256)["color"]
    on = Simulation(p_on, gpu).render_maps(256)["color"]
    assert np.allclose(base, on, atol=1e-3), np.abs(base - on).max()


def test_cirrus_localized_to_cloud_windows(gpu):
    """Pinned configuration: intermittency 1e-6 in BOTH renders (same FX
    program, identical gate path), only cirrus_fibers varies. The q >= 2.2
    early-out adds exactly 0.0 outside the windows, so far pixels are
    byte-equal; inside they differ."""
    base = _synth_detail_field(gpu, _cloud_params(intermittency=1e-6))
    on = _synth_detail_field(
        gpu, _cloud_params(intermittency=1e-6, cirrus_fibers=1.4)
    )
    sim = Simulation(_cloud_params(), gpu)
    from gasgiant.engine.snapshot import bright_cloud_centers

    assert bright_cloud_centers(sim.vortices), (
        "seed 42 with companions+accent must produce bright clouds"
    )
    inside = _cloud_windows(sim, base.shape)

    np.testing.assert_array_equal(base[~inside], on[~inside])
    assert not np.array_equal(base[inside], on[inside])
    assert np.all(np.isfinite(on))


def test_cirrus_frequency_dial_is_live(gpu):
    lo = _synth_detail_field(
        gpu, _cloud_params(intermittency=1e-6, cirrus_fibers=1.4,
                           cirrus_fiber_freq=4.0)
    )
    hi = _synth_detail_field(
        gpu, _cloud_params(intermittency=1e-6, cirrus_fibers=1.4,
                           cirrus_fiber_freq=8.0)
    )
    sim = Simulation(_cloud_params(), gpu)
    inside = _cloud_windows(sim, lo.shape)
    assert not np.array_equal(lo[inside], hi[inside])
    # Identical everywhere else (the dial only acts inside the windows).
    np.testing.assert_array_equal(lo[~inside], hi[~inside])


def test_cirrus_render_is_seam_clean(gpu):
    sim = Simulation(_cloud_params(cirrus_fibers=1.2), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems


def test_new_cirrus_knobs_are_post_tier():
    from gasgiant.engine.invalidation import diff_tiers
    from gasgiant.params.model import Tier

    a = PlanetParams()
    b = a.model_copy(deep=True)
    b.detail.cirrus_fibers = 0.8
    b.detail.cirrus_fiber_freq = 12.0
    b.detail.streak_mute = 1.0
    assert diff_tiers(a, b) == {Tier.POST}
