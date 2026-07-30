"""The baroclinic artist levers: the storm band, and the two seeding levers.

The baroclinic source never grows -- eddy variance decays monotonically at every
resolution and every affordable configuration (falsification record:
docs/superpowers/specs/2026-07-30-baroclinic-artist-levers-design.md). What
reaches the solver is therefore the SEED, and the seed had exactly two properties
the eye reads as a mechanical comb:

    one wavelength      every crest the same width and spacing
    one global phase    every crest aligned pole-to-pole

`spectrum_width` and `phase_jitter` address those two defects respectively, and
the point of this module is that they are ORTHOGONAL: each moves its own metric
and leaves the other's alone. If that ever stops holding, one lever has started
doing the other's job and the artist has lost independent control.

Metrics deliberately avoid "phase concentration of the argmax mode", which is not
trustworthy -- it reports the phase of whichever mode happens to win, and that can
be the C-grid grid-scale artifact rather than the seeded one. Everything here is
measured at a FIXED seeded wavenumber.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref

BASE = dict(W=bsrc.SRC_W, H=bsrc.SRC_H, unstable=True, seed=4201,
            gp1=bsrc.GP1, gp2=bsrc.GP2, xi_unstable=bsrc.XI,
            m_zonal=bsrc.M_ZONAL, pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0)
STATE_FIELDS = ("h1", "h2", "u1", "v1", "u2", "v2")


def _state(**over):
    return ref.baroclinic_test_state(**{**BASE, **over})


def _source(**over) -> np.ndarray:
    """The field the solver actually receives.

    Metrics are taken here rather than on raw ``h2`` deliberately: ``h2`` also
    carries the broadband seed noise, which is white and drags any spectral
    statistic toward the grid scale (its power centroid reads ~26 even for a pure
    single-mode seed). The smoothing inside the source derivation is what makes
    the seeded structure the dominant signal, so that is where "does the artist
    see one wavelength" can honestly be measured.
    """
    return bsrc.geostrophic_vorticity_source(
        _state(**over), smooth_sigma=bsrc.SMOOTH_SIGMA)


def _active_rows(field: np.ndarray) -> np.ndarray:
    """Rows carrying real signal, so near-silent rows cannot dominate a mean."""
    amp = np.abs(field).sum(axis=1)
    return field[amp > 0.15 * amp.max()]


def _phase_alignment(field: np.ndarray, m: int) -> float:
    """Amplitude-weighted cross-latitude phase concentration at a FIXED m.
    1.0 = every row's crest at the same longitude (the comb); 0 = uncorrelated."""
    rows = _active_rows(field)
    spec = np.fft.rfft(rows - rows.mean(axis=1, keepdims=True), axis=1)
    a = np.abs(spec[:, m])
    return float(np.abs((a * np.exp(1j * np.angle(spec[:, m]))).sum() / a.sum()))


def _dominant_share(field: np.ndarray) -> float:
    """Fraction of zonal power sitting in the single strongest mode.
    High = one wavelength repeated; low = a spread of spacings."""
    rows = _active_rows(field)
    p = (np.abs(np.fft.rfft(rows - rows.mean(axis=1, keepdims=True), axis=1)) ** 2)
    w = p[:, 1:].sum(axis=0)
    return float(w.max() / w.sum())


# -- no-op (byte identity) ----------------------------------------------------


@pytest.mark.parametrize("over", [
    {},
    {"phase_jitter": 0.0},
    {"spectrum_width": 0},
    {"phase_jitter": 0.0, "spectrum_width": 0},
])
def test_levers_off_are_bitwise_identical(over):
    """Both default to a no-op, and the guard is structural: with them off the
    original single-mode expression is the only thing that runs, so the result is
    bit-for-bit unchanged rather than merely algebraically equal."""
    a, b = _state(), _state(**over)
    for f in STATE_FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert a.dt == b.dt


def test_band_levers_at_their_defaults_are_bitwise_identical():
    """Promoting the band constants to params must not move the default render."""
    a = _state()
    b = _state(phi_test_deg=bsrc.PHI_TEST_DEG,
               band_halfwidth_deg=bsrc.BAND_HALFWIDTH_DEG)
    for f in STATE_FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


# -- each lever does its own job ---------------------------------------------


def test_phase_jitter_breaks_crest_alignment():
    off = _phase_alignment(_source(), bsrc.M_ZONAL)
    on = _phase_alignment(_source(phase_jitter=2.0), bsrc.M_ZONAL)
    assert off > 0.9, f"the default really is a pole-to-pole comb (got {off})"
    assert on < 0.5, f"jitter must decorrelate the crests (got {on})"


def test_spectrum_width_breaks_the_single_wavelength():
    off = _dominant_share(_source())
    on = _dominant_share(_source(spectrum_width=4))
    assert off > 0.5, f"the default really is one wavelength (got {off})"
    assert on < 0.5 * off, f"the packet must spread the power (got {on})"


def test_spectrum_width_preserves_feature_size():
    """Varying the spacing must not silently resize the storms -- the centroid
    stays at the seeded wavenumber, only the spread around it grows."""
    rows = _active_rows(_source(spectrum_width=4))
    p = (np.abs(np.fft.rfft(rows - rows.mean(axis=1, keepdims=True), axis=1)) ** 2)
    w = p[:, 1:].sum(axis=0)
    ms = np.arange(1, p.shape[1])
    centroid = float((w * ms).sum() / w.sum())
    assert abs(centroid - bsrc.M_ZONAL) < 3.0, centroid


# -- orthogonality ------------------------------------------------------------


def test_the_two_levers_are_independent():
    """Each moves its own metric and leaves the other's where it was. This is
    what lets an artist reach for one without disturbing the other."""
    base = _source()
    jit = _source(phase_jitter=2.0)
    spec = _source(spectrum_width=4)

    # jitter: alignment falls, spectral concentration barely moves
    assert _phase_alignment(jit, bsrc.M_ZONAL) < 0.5
    assert abs(_dominant_share(jit) - _dominant_share(base)) < 0.15

    # spectrum: concentration falls, alignment barely moves
    assert _dominant_share(spec) < 0.5 * _dominant_share(base)
    assert abs(_phase_alignment(spec, bsrc.M_ZONAL)
               - _phase_alignment(base, bsrc.M_ZONAL)) < 0.15


def test_levers_do_not_perturb_the_broadband_noise_realisation():
    """Each lever draws from its OWN named substream. If either shared the
    generator that produces the broadband seed noise, turning it on would
    reshuffle that field too and the lever would not be an isolated axis.

    Probed via the ZONAL-MEAN interface profile, which the levers cannot touch
    (both edit a zero-zonal-mean longitudinal pattern) but a reshuffled noise
    draw would move.
    """
    base = _state().h2.mean(axis=1)
    for over in ({"phase_jitter": 2.0}, {"spectrum_width": 4},
                 {"phase_jitter": 2.0, "spectrum_width": 4}):
        assert np.allclose(_state(**over).h2.mean(axis=1), base, rtol=0, atol=1e-9), over


# -- the storm band construction ---------------------------------------------


@pytest.mark.parametrize("latitude", [20.0, 30.0, 45.0, 60.0, 75.0])
def test_the_band_builds_without_clipping_anywhere_in_its_declared_range(latitude):
    """A slider must not promise a range it cannot deliver.

    The base state sets h1 = H1_mean - A*cumint with A = xi*H2_mean/tan(lat), so
    the interface swing diverges as the band moves equatorward -- it nearly
    doubles between 45 and 28 degrees. Before the upper layer was allowed to
    deepen, everything at or below ~35 degrees clipped h1 to the floor, which
    breaks the geostrophic balance the base state exists to satisfy; the warmup
    then outcropped and the feature silently switched itself off partway down
    a slider an artist was dragging.
    """
    st = ref.baroclinic_test_state(
        **{**BASE, "phi_test_deg": latitude,
           "band_halfwidth_deg": min(25.0, latitude * 0.55)})
    assert float(st.h1.min()) > 1.0, "upper layer clipped at build"
    assert float(st.h2.min()) > 1.0, "lower layer clipped at build"


def test_deepening_is_inert_at_the_default_band():
    """The deepening must not fire where it is not needed, or it would move the
    default render. At the default construction h1 clears the floor already."""
    st = _state()
    assert float(st.h1.min()) > 1.0
    assert st._H_mean == pytest.approx(25000.0, abs=1e-9), (
        "no deepening happened, so the realized depth is the requested one")


# -- the equator clamp --------------------------------------------------------


def test_the_band_is_clamped_off_the_equator():
    """A band straddling f = 0 is not a configuration to support.

    Both the source proxy (gp2/f) and the base-state swing (~cot latitude)
    diverge at the equator, and the LOWER layer clips at build -- which the
    upper-layer deepening does not address. Measured: every (latitude, width)
    combination fails once width reaches latitude.
    """
    from gasgiant.params.model import baroclinic_effective_width as eff
    # Inert where the band already clears the equator, including the default.
    assert eff(45.0, 25.0) == 25.0
    assert eff(75.0, 40.0) == 40.0
    # Clamped where it would not.
    assert eff(20.0, 40.0) == 15.0
    assert eff(30.0, 40.0) == 25.0
    # Never below the narrowest band the source grid can represent, even at the
    # extreme corner of the declared ranges.
    from gasgiant.params.model import BAROCLINIC_MIN_WIDTH_DEG
    assert eff(20.0, 40.0) >= BAROCLINIC_MIN_WIDTH_DEG


@pytest.mark.slow          # 8 x 2000 CPU reference-solver steps, ~96 s total
@pytest.mark.parametrize("latitude,width", [
    (20.0, 8.0), (20.0, 40.0), (30.0, 40.0), (45.0, 8.0),
    (45.0, 25.0), (45.0, 40.0), (75.0, 8.0), (75.0, 40.0),
])
def test_every_declared_band_corner_builds_and_warms(latitude, width):
    """The clamp must map the WHOLE declared (latitude, width) rectangle into the
    measured-usable region -- otherwise a slider still fails partway along."""
    from gasgiant.params.model import baroclinic_effective_width
    eff = baroclinic_effective_width(latitude, width)
    st = ref.baroclinic_test_state(
        **{**BASE, "phi_test_deg": latitude, "band_halfwidth_deg": eff})
    assert float(st.h1.min()) > 1.0, "upper layer clipped at build"
    assert float(st.h2.min()) > 1.0, "lower layer clipped at build"
    for _ in range(2000):
        ref.step_2layer(st)          # PositivityViolation here fails the test

    # Surviving the solver is NOT enough, and asserting only that is what let a
    # washed-out band at latitude 15 (dominant m=2, 2.3% of power at the seeded
    # mode) sit inside the declared range. Check what the driver would actually
    # ship: the coherence gate must pass AND the seeded mode must still dominate.
    band, taper = bsrc.mask_band_for(latitude, eff)
    zeta = bsrc.geostrophic_vorticity_source(
        st, smooth_sigma=bsrc.SMOOTH_SIGMA, lat_band=band, taper=taper)
    # in_band mirrors production: the fixed-window gate samples latitudes
    # 53.4..15.9 and does not follow a steered band, so it grades empty rows.
    m = bsrc.assert_coherent(zeta, in_band=True)   # raises -> test fails
    assert m == bsrc.M_ZONAL, f"dominant mode drifted off the seeded one: {m}"


def test_the_clamp_is_reported_not_silent():
    """An artist who asked for a wider belt than they got must be told."""
    from gasgiant.params.model import PlanetParams, SolverType
    p = PlanetParams()
    p.solver.type = SolverType.VORTICITY
    p.solver.baroclinic.enabled = True
    p.solver.baroclinic.latitude = 20.0
    p.solver.baroclinic.width = 40.0
    assert any("across the equator" in w for w in p.validation_warnings())

    p.solver.baroclinic.latitude = 45.0
    p.solver.baroclinic.width = 25.0
    assert not any("across the equator" in w for w in p.validation_warnings())


# -- pins on the two output-changing bugs found in review ---------------------


def test_mask_band_default_reproduces_the_historical_hardcoded_mask():
    """`mask_band_for` must return EXACTLY the mask the source used before it was
    parameterised, or enabling baroclinic renders differently than it did.

    Sizing the pad from `taper` (8) instead of MASK_PAD_DEG (10) gives (12, 78)
    and moves the default source by max 0.885 on a unit-std field -- and nothing
    else catches it: the metric helpers here call
    `geostrophic_vorticity_source` with its own default lat_band and never route
    through `mask_band_for`, the GPU m3 gates are inequality-based, and p05 is
    vacuous because no preset enables baroclinic.
    """
    assert bsrc.mask_band_for(bsrc.PHI_TEST_DEG, bsrc.BAND_HALFWIDTH_DEG) == (
        (10.0, 80.0), 8.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
def test_spectrum_width_holds_the_injected_amplitude_fixed(k):
    """`spectrum_width` must vary SPACING only. Independent phases give
    var(sum) = sum(amp^2)/2 and a lone cosine has var 1/2, so the divisor is
    sqrt(sum(amp^2)); using sqrt(0.5*sum(amp^2)) normalizes to unit variance and
    silently makes this a 1.41x amplitude lever too -- which would also eat the
    184 m of build-time upper-layer headroom the default construction has.
    """
    lam = (np.arange(bsrc.SRC_W) + 0.5) * (2.0 * np.pi / bsrc.SRC_W)
    single = float(np.cos(bsrc.M_ZONAL * lam).std())
    packet = float(ref._seed_pattern(
        lam, bsrc.SRC_H, bsrc.SRC_W, bsrc.M_ZONAL, 0.0, 0.0, k, 0).std())
    assert packet == pytest.approx(single, rel=1e-9)


def test_band_defaults_agree_across_all_three_homes():
    """45/25 lives in the params defaults, in `baroclinic_source` (the driver's
    None fallback) and in `shallow_water_ref` (the `phi_test_deg=None` fallback).
    Only params <-> bsrc was pinned; edit the `ref` privates and the driver would
    silently keep 45 while every direct `baroclinic_test_state()` caller moved.
    """
    from gasgiant.params.model import PlanetParams
    b = PlanetParams().solver.baroclinic
    assert (ref._PHI_TEST_DEG, ref._BAND_HALFWIDTH_DEG) == (
        bsrc.PHI_TEST_DEG, bsrc.BAND_HALFWIDTH_DEG) == (b.latitude, b.width)


def test_the_coherence_gate_follows_a_steered_band():
    """The production gate must measure the band, not a fixed latitude window.

    `dominant_zonal_m`'s default samples rows 19..40 of 96 -- latitudes
    53.4..15.9 -- which was correct while the band was hardcoded at 45 +/- 25 and
    is wrong the moment `latitude` becomes a slider. A band at 75 +/- 8 spans
    67..83 and lies ENTIRELY outside that window, so the fixed gate reads empty
    rows and reports a meaningless m=1 while the band itself is clean.
    """
    st = _state(phi_test_deg=75.0, band_halfwidth_deg=8.0)
    band, taper = bsrc.mask_band_for(75.0, 8.0)
    zeta = bsrc.geostrophic_vorticity_source(
        st, smooth_sigma=bsrc.SMOOTH_SIGMA, lat_band=band, taper=taper)

    fixed, _ = bsrc.dominant_zonal_m(zeta)
    in_band, _ = bsrc.dominant_zonal_m_in_band(zeta)
    assert fixed != bsrc.M_ZONAL, (
        "if the fixed window ever starts covering this band, this test has "
        "stopped proving anything")
    assert in_band == bsrc.M_ZONAL, in_band


def test_band_aware_and_fixed_gates_agree_on_the_default_band():
    """The band-aware path must not move the DEFAULT gate verdict -- that is what
    keeps `in_band=True` from being a behaviour change for the shipped config."""
    st = _state()
    band, taper = bsrc.mask_band_for(bsrc.PHI_TEST_DEG, bsrc.BAND_HALFWIDTH_DEG)
    zeta = bsrc.geostrophic_vorticity_source(
        st, smooth_sigma=bsrc.SMOOTH_SIGMA, lat_band=band, taper=taper)
    assert bsrc.dominant_zonal_m(zeta)[0] == bsrc.dominant_zonal_m_in_band(zeta)[0]
