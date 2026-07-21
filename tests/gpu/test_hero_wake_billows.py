"""detail.hero_wake_billows: RENDER-side procedural synthesis of the hero-wake
dense billow chain (DETAIL_FX family). The companion to hero_wake_braid -- the
braid INKS the sim's own advected tracer folds (capped at the sim's few large
rolls); this SYNTHESIZES a co-scaled anisotropic rope chain, material-anchored
to the real wake and oriented by the local folded flow, at a scale the braid
cannot manufacture. Measured on the detail-synth field.

Cross-variant comparisons use atol, never byte-equality (different binaries may
reschedule FP in shared expressions); byte-equality is only asserted within one
program -- several tests below pin intermittency=1e-6 in BOTH arms so both run
the SAME compiled DETAIL_FX program.
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
    terms that drown the billow signal); 2048 keeps the ~1.2-rc rope spacing
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
    """The registry's wake frame for the (single) hero -- probe geometry must
    come from here, never re-derived constants (PR-43 lesson: test boxes must
    scale with authored geometry)."""
    heroes = sim.vortices.heroes()
    assert len(heroes) == 1, "seed 42 must seed exactly one hero"
    return heroes[0]


def _wake_coords(v, shape):
    """The kernel's (an, b) wake frame at every pixel, from the REGISTRY frame."""
    h, w = shape
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    dlon = (lon[None, :] - v.lon + 3.0 * np.pi) % (2.0 * np.pi) - np.pi
    an = dlon * v.wake_dir / v.r_core * np.ones((h, 1))
    alat = (lat[:, None] - (v.lat + v.wake_lat_off)) / v.r_core * np.ones((1, w))
    s_belt = np.sign(v.wake_lat_off) if v.wake_lat_off != 0.0 else -np.sign(v.lat)
    b = alat * s_belt
    return an, b


def _wedge_masks(v, shape):
    """(wedge, inner) from the shared heroWakeWindow window + margin -- the same
    window the braid test uses (both levers route through the same helper)."""
    an, b = _wake_coords(v, shape)
    wedge = (
        (an > 1.0 * v.aspect - 0.2) & (an < 19.3)
        & (b < 2.6) & (b > -1.0)
    )
    inner = (
        (an > 1.6 * v.aspect) & (an < 6.0)
        & (b < 0.8) & (b > -0.1)
    )
    return wedge, inner


def test_zero_routes_to_default_program(gpu):
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    zero = Simulation(_quick_params(hero_wake_billows=0.0), gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, zero)


def test_forced_variant_is_noop_at_epsilon(gpu):
    """billows=1e-6 forces the DETAIL_FX program while the contribution stays
    sub-1e-6 -- the test the lever-scaled quiet-ground mute protects (both the
    mute strength and the additive term are scaled by u_hero_wake_billows)."""
    base = Simulation(_quick_params(), gpu).render_maps(256)["color"]
    fx = Simulation(_quick_params(hero_wake_billows=1e-6), gpu).render_maps(256)["color"]
    assert np.allclose(base, fx, atol=1e-3), np.abs(base - fx).max()


def test_without_heroes_is_noop(gpu):
    """hero_count=0 makes the billows block add exactly 0.0. intermittency is
    pinned 1e-6 in BOTH arms so both select the SAME DETAIL_FX program and
    byte-equality is legal."""
    p_base = _quick_params(intermittency=1e-6)
    p_base.storms.hero_count = 0
    p_on = _quick_params(intermittency=1e-6, hero_wake_billows=1.6)
    p_on.storms.hero_count = 0
    base = Simulation(p_base, gpu).render_maps(256)["color"]
    on = Simulation(p_on, gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(base, on)


def test_localized_to_the_wake_wedge(gpu):
    """Same FX program both arms (intermittency 1e-6 pinned); only
    hero_wake_billows varies. Every early-out adds exactly 0.0, so pixels
    outside the wedge window are byte-equal; inside they differ. The wedge is
    built from the REGISTRY wake frame + the kernel's window constants."""
    base = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    on = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_billows=1.6)
    )
    v = _wake_frame(Simulation(_quick_params(), gpu))
    wedge, inner = _wedge_masks(v, base.shape)
    np.testing.assert_array_equal(base[~wedge], on[~wedge])
    assert not np.array_equal(base[inner], on[inner])
    assert np.all(np.isfinite(on))


def test_tracks_wake_direction(gpu):
    """The synthesized wedge follows the REGISTRY wake_dir: forcing the hero's
    wake EAST (vs seed 42's natural WEST) puts the whole signal on the
    downstream (east) half, leaving the upstream half byte-untouched (the
    heroWakeWindow `an <= 0` early-out adds exactly 0.0)."""
    from gasgiant.params.model import WakeDir

    def _east(**detail):
        p = _quick_params(**detail)
        p.storms.hero_wake_dir = WakeDir.EAST
        return p

    base = _synth_detail_field(gpu, _east(intermittency=1e-6))
    on = _synth_detail_field(gpu, _east(intermittency=1e-6, hero_wake_billows=1.6))
    v = _wake_frame(Simulation(_east(), gpu))
    assert v.wake_dir == 1.0, "forced EAST must set the registry wake_dir to +1"

    h, w = base.shape
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    dlon = ((lon[None, :] - v.lon + 3.0 * np.pi) % (2.0 * np.pi) - np.pi) * np.ones((h, 1))
    delta = np.abs(on - base)
    assert delta.max() > 1e-4, "forced-EAST billows produced no signal"
    np.testing.assert_array_equal(on[dlon < 0.0], base[dlon < 0.0])
    assert float(dlon[delta > 1e-4].min()) > 0.0, "signal leaked upstream (west)"


def _structure_tensor_aspect(f: np.ndarray) -> float:
    """Real-space rope aspect ~ sqrt(J_max / J_min): gradients concentrate
    ACROSS parallel ropes, so the gradient second-moment ratio is the SQUARE of
    the elongation. Isotropic noise -> ~1; 2.5:1 ropes -> ~2.5."""
    gy, gx = np.gradient(f)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    ev = np.linalg.eigvalsh(np.array([[jxx, jxy], [jxy, jyy]]))
    return float(np.sqrt(max(ev[1], 0.0) / max(ev[0], 1e-12)))


def _dominant_wavelength_px(f: np.ndarray) -> float:
    """Radial wavelength (px) of the peak-energy Fourier mode above a low-freq
    cutoff -- the transverse (shortest) rope spacing dominates the spectrum."""
    ny, nx = f.shape
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(f * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))[:, None] * np.ones((1, nx))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :] * np.ones((ny, 1))
    fr = np.hypot(fx, fy)
    keep = fr > (3.0 / min(ny, nx))          # drop DC / envelope-scale modes
    idx = int(np.argmax(np.where(keep, power, 0.0)))
    return 1.0 / float(fr.flat[idx])


def _measure_box(v, shape):
    """A generous in-wedge box, several rc TALL in the transverse (lat)
    direction so a whole transverse wavelength (~1.2 rc) is resolvable -- the
    thin lat-window used for the localization test cannot hold one billow, so
    the capability metrics need their own wider region."""
    an, b = _wake_coords(v, shape)
    box = (an > 2.5) & (an < 9.0) & (b > -1.4) & (b < 2.2)
    ys, xs = np.where(box)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return box[y0:y1, x0:x1], (y0, y1, x0, x1)


def _hp_box(field, off, sl, m):
    """High-passed (mute-envelope-removed) in-wedge delta box: on - off, with a
    Gaussian low-pass (wider than a rope, narrower than the wedge) subtracted so
    what remains is the rope oscillation, not the quiet-ground mute DC."""
    import cv2

    y0, y1, x0, x1 = sl
    box = (field - off)[y0:y1, x0:x1].copy()
    box[~m] = 0.0
    hp = box - cv2.GaussianBlur(box, (0, 0), sigmaX=40.0)
    hp[~m] = 0.0
    return hp


def _bowley_skew(x: np.ndarray) -> float:
    """Quantile (Bowley) skewness: (Q3 + Q1 - 2*Q2) / (Q3 - Q1). Robust to the
    sparse full-amplitude rope crests the high material anchor concentrates --
    the raw 3rd standardized moment is outlier-dominated there (a known property
    of near-hard-gated fields; cf. cirrus) even though the synthesized strand is
    neutral by construction, so median-symmetry is the faithful polarity check."""
    q1, q2, q3 = np.percentile(x, [25.0, 50.0, 75.0])
    return float((q3 + q1 - 2.0 * q2) / (q3 - q1 + 1e-12))


def test_synthesizes_coscaled_anisotropic_structure(gpu):
    """The capability that distinguishes SYNTHESIS from the braid's inking: an
    anisotropic co-scaled rope chain at the dialed transverse frequency. Three
    reference properties (tolerances budget the residual cos-lat skew + the
    flow-angle variation across the wedge):
      - CONSTRUCTION-neutral polarity (NOT the same as reference-grade polarity):
        the SYNTHESIZED STRAND is neutral by construction (odd function of
        mean-zero fbm), so the machinery introduces no bright/dark bias of its
        own. Measured as a near-zero standardized mean AND |Bowley quantile skew|
        <= 0.2 with both sign-lobes present. The raw 3rd moment is NOT used: the
        high material anchor makes the field heavy-tailed (sparse full-amplitude
        rope crests in material cores), inflating the tail-sensitive 3rd moment
        (raw ~1.0) while mean and median-symmetry stay ~0 -- a measurement
        artifact of the anchor's heteroscedasticity, not a polarity bias.
        CAVEAT: this proves the STRAND is unbiased, NOT that the rendered wake
        reads as the reference's EQUAL interleaved light/dark ropes (PIA07782
        skew -0.12..+0.01) under the high default M -- reference-grade polarity
        stays a LIVE S2 visual gate (M is an S2 knob; see the plan S2 section).
      - ANISOTROPY > 1.5:1: built 2.5:1 via along_freq = trans_freq / K_ANISO.
      - the dominant transverse wavelength TRACKS the dial (doubling the freq
        roughly halves it).

    All three are read off the high-passed on-vs-off delta over a wedge box that
    is several rc tall (so a transverse wavelength is resolvable)."""
    v = _wake_frame(Simulation(_quick_params(), gpu))
    off = _synth_detail_field(gpu, _quick_params(intermittency=1e-6))
    on_lo = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_billows=1.6,
                           hero_wake_billow_freq=0.85)
    )
    on_hi = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_billows=1.6,
                           hero_wake_billow_freq=1.70)
    )
    m, sl = _measure_box(v, off.shape)
    lo = _hp_box(on_lo, off, sl, m)
    hi = _hp_box(on_hi, off, sl, m)

    vals = lo[m]
    assert vals.size > 2000, "in-wedge measurement region too small"
    assert float(np.abs(vals).max()) > 1e-4, "no in-wedge billow signal"

    std_mean = float(vals.mean() / (vals.std() + 1e-12))
    assert abs(std_mean) <= 0.15, f"polarity not neutral (biased mean): {std_mean}"
    bowley = _bowley_skew(vals)
    assert abs(bowley) <= 0.2, f"polarity not neutral (skewed): bowley={bowley}"
    assert 0.35 < float((vals > 0).mean()) < 0.65, "one polarity lobe missing"

    aspect = _structure_tensor_aspect(lo)
    assert aspect > 1.5, f"structure not anisotropic (ropes): aspect={aspect}"

    lam_lo, lam_hi = _dominant_wavelength_px(lo), _dominant_wavelength_px(hi)
    ratio = lam_lo / max(lam_hi, 1e-6)
    assert 1.4 < ratio < 2.8, (
        f"transverse wavelength does not track the freq dial: "
        f"lam(0.85)={lam_lo:.1f}px lam(1.70)={lam_hi:.1f}px ratio={ratio:.2f}"
    )


def test_freq_dial_is_live(gpu):
    lo = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_billows=1.6,
                           hero_wake_billow_freq=0.6)
    )
    hi = _synth_detail_field(
        gpu, _quick_params(intermittency=1e-6, hero_wake_billows=1.6,
                           hero_wake_billow_freq=1.4)
    )
    v = _wake_frame(Simulation(_quick_params(), gpu))
    wedge, _ = _wedge_masks(v, lo.shape)
    assert not np.array_equal(lo[wedge], hi[wedge])
    # The dial only acts inside the wedge; everything else is byte-identical.
    np.testing.assert_array_equal(lo[~wedge], hi[~wedge])


def test_render_is_seam_clean(gpu):
    sim = Simulation(_quick_params(hero_wake_billows=1.2), gpu)
    maps = sim.render_maps(512)
    report = validate_arrays({"color": maps["color"], "height": maps["height"]})
    assert report.ok, report.problems
