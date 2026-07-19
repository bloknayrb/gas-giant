"""detail.hero_wake_braid: RENDER-side braided rope texture in the hero wake
(DETAIL_FX family). NOT storms.hero_wake_detail — that is the SIM lever that
churns the tracers in the wake (tests/gpu/test_hero_wake_detail.py); this one
synthesizes fine crossing rope structure at render time on top of it, measured
on the detail-synth field.

Cross-variant comparisons use atol, never byte-equality (different binaries
may reschedule FP in shared expressions); byte-equality is only asserted
within one program — several tests below pin intermittency=1e-6 in BOTH arms
so both run the SAME compiled DETAIL_FX program.
"""
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
    for key, value in detail.items():
        setattr(p.detail, key, value)
    return p


def _synth_detail_field(gpu, params, size=(2048, 1024)) -> np.ndarray:
    """DetailSynth directly (the composed color map mixes in cells + tracer
    terms that drown the braid signal); 2048 keeps the ~0.82-rc rope spacing
    well above the pixel-resolvability atten."""
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


def _wake_frame(sim):
    """The registry's wake frame for the (single) hero — probe geometry must
    come from here, never re-derived constants (PR-43 lesson: test boxes must
    scale with authored geometry)."""
    heroes = sim.vortices.heroes()
    assert len(heroes) == 1, "seed 42 must seed exactly one hero"
    return heroes[0]


def test_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(hero_wake_braid=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_forced_fx_variant_is_noop_at_epsilon(gpu):
    """braid=1e-6 forces the DETAIL_FX program while the contribution stays
    sub-1e-6 — the test that actually exercises the variant text."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_quick_params(hero_wake_braid=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


def test_without_heroes_is_noop_on_the_fx_program(gpu):
    """hero_count=0 makes the braid block add exactly 0.0. intermittency is
    pinned 1e-6 in BOTH arms so both select the SAME DETAIL_FX program and
    byte-equality is legal (a braid-off vs braid-on comparison across
    variants would be the illegal cross-variant byte assert)."""
    p_base = _quick_params(intermittency=1e-6)
    p_base.storms.hero_count = 0
    p_braid = _quick_params(intermittency=1e-6, hero_wake_braid=1.5)
    p_braid.storms.hero_count = 0
    base = Simulation(p_base, gpu).render_maps(256)["color"]
    braid = Simulation(p_braid, gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, braid)


def test_braid_localized_to_the_wake_wedge(gpu):
    """Same FX program both arms (intermittency 1e-6 pinned); only
    hero_wake_braid varies. Every early-out in the block adds exactly 0.0,
    so pixels outside the wedge window are byte-equal; inside they differ.
    The wedge is built from the REGISTRY wake frame + the kernel's window
    constants with margin."""
    base = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    braid = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_braid=1.5)
    )
    sim = Simulation(_quick_params(), gpu)
    v = _wake_frame(sim)

    h, w = base.shape
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    dlon = (lon[None, :] - v.lon + 3.0 * np.pi) % (2.0 * np.pi) - np.pi
    an = dlon * v.wake_dir / v.r_core * np.ones((h, 1))
    alat = (lat[:, None] - (v.lat + v.wake_lat_off)) / v.r_core * np.ones((1, w))
    s_belt = np.sign(v.wake_lat_off) if v.wake_lat_off != 0.0 else -np.sign(v.lat)
    b = alat * s_belt
    # Kernel windows: an in (1.0*asp, 19.0), beltward < 2.4 rc, zoneward
    # < 0.80 rc — dilate by margin so smoothstep edges stay inside "wedge".
    wedge = (
        (an > 1.0 * v.aspect - 0.2) & (an < 19.3)
        & (b < 2.6) & (b > -1.0)
    )
    inner = (
        (an > 1.6 * v.aspect) & (an < 6.0)
        & (b < 0.8) & (b > -0.1)
    )
    np.testing.assert_array_equal(base[~wedge], braid[~wedge])
    assert not np.array_equal(base[inner], braid[inner])
    assert np.all(np.isfinite(braid))


def test_braid_inks_the_tracer_folds(gpu):
    """REV 4: the signal is KEYED to the tracer field (pale entrained
    material brightened, fold-boundary gradients inked dark), not a
    synthetic carrier. So inside the wedge the delta must correlate with
    the tracer structure: negative (inked) pixels sit at higher local T0
    gradient than positive ones, and the delta is nonzero where the wake
    stamp puts structure. Generous — S2 owns the calibration metrics."""
    base = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    braid = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_braid=1.5)
    )
    sim = Simulation(_quick_params(), gpu)
    _wake_frame(sim)  # asserts the single-hero premise
    delta = braid - base
    assert float(np.abs(delta).max()) > 1e-4

    # T0 field at synth resolution for gradient comparison.
    t0 = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]
    import cv2
    t0r = cv2.resize(t0, (delta.shape[1], delta.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    gy, gx = np.gradient(t0r)
    gmag = np.hypot(gx, gy)

    inked = delta < -0.02
    lifted = delta > 0.02
    # Both polarities must be present, else the discriminating assertion below
    # is vacuously skipped and a mis-keyed-but-nonzero signal passes (the two
    # populations ARE the mechanism: rims darken, cores brighten).
    assert inked.sum() > 50, "no inked rim pixels in the wedge"
    assert lifted.sum() > 50, "no lifted (brightened core) pixels in the wedge"
    assert float(gmag[inked].mean()) > float(gmag[lifted].mean()), (
        "inked pixels should sit on fold boundaries (high T0 gradient)"
    )


def test_braid_wedge_tracks_wake_direction(gpu):
    """The inked wedge follows the REGISTRY wake_dir, not a hardcoded side:
    forcing the hero's wake EAST (vs seed 42's natural WEST) puts the whole
    inked region on the opposite along-wake half, leaving the upstream half
    byte-untouched. This guards the registry-derived direction plumbing across
    a chirality-style wake flip -- the lever was rebased across exactly such a
    flip (which moved warm's wake E->W), so an EAST-forced case that no other
    test exercises is the regression this needs."""
    from gasgiant.params.model import WakeDir

    def _east(**detail):
        p = _quick_params(**detail)
        p.storms.hero_wake_dir = WakeDir.EAST
        return p

    base = _synth_detail_field(gpu, _east(intermittency=1e-6))
    braid = _synth_detail_field(gpu, _east(intermittency=1e-6, hero_wake_braid=1.5))
    v = _wake_frame(Simulation(_east(), gpu))
    assert v.wake_dir == 1.0, "forced EAST must set the registry wake_dir to +1"

    h, w = base.shape
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    dlon = ((lon[None, :] - v.lon + 3.0 * np.pi) % (2.0 * np.pi) - np.pi) * np.ones((h, 1))
    delta = np.abs(braid - base)
    assert delta.max() > 1e-4, "forced-EAST braid produced no signal"
    # Upstream is WEST here (dlon < 0): the kernel's `an <= 0` early-out adds
    # exactly 0.0 there, so the entire upstream half is byte-identical...
    np.testing.assert_array_equal(braid[dlon < 0.0], base[dlon < 0.0])
    # ...and every changed pixel sits DOWNSTREAM = EAST (dlon > 0).
    assert float(dlon[delta > 1e-4].min()) > 0.0, "ink leaked upstream (west)"


def test_braid_render_is_seam_clean(gpu):
    sim = Simulation(_quick_params(hero_wake_braid=1.2), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems
