"""GPU: bands.belt_fade / bands.faded_band_index + outbreak placement (W6).

Byte-identity discipline per CLAUDE.md: the kinematic path asserts EXACT
equality for the forced-variant no-op; the vorticity path asserts within the
documented session-noise floor (GPU_NOISE_ATOL = 1e-2 -- its SOR Poisson
solve is not byte-stable across GL session contexts).
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.events import KIND_OUTBREAK, TRAIN_LAT_SPREAD

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2  # vorticity SOR session-context floor (tests/gpu/test_checkpoint.py)


def _kinematic(**bands) -> PlanetParams:
    p = PlanetParams(seed=21)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 0  # keep T0 statistics band-driven
    for key, value in bands.items():
        setattr(p.bands, key, value)
    return p


def _vorticity(**bands) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    for key, value in bands.items():
        setattr(p.bands, key, value)
    return p


def _t0(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


def _band_rows(t0: np.ndarray, lat_hi: float, lat_lo: float, pad_frac: float = 0.3):
    """Row mask for the band interior (padded away from the smoothed edges)."""
    h = t0.shape[0]
    lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    pad = pad_frac * (lat_hi - lat_lo)
    return (lats < lat_hi - pad) & (lats > lat_lo + pad)


# ---------------------------------------------------------------- no-op renders

def test_forced_noop_kinematic_byte_identical(gpu):
    """belt_fade=0.0 + faded_band_index pinned to the heuristic's own pick is
    byte-identical to the default render on the kinematic path."""
    p_base = _kinematic()
    base_layout = generate_bands(p_base.seed, p_base.bands)
    base = _t0(Simulation(p_base, gpu))
    forced = _t0(Simulation(
        _kinematic(belt_fade=0.0, faded_band_index=int(base_layout.fade_index)), gpu))
    np.testing.assert_array_equal(forced, base)


def test_forced_noop_vorticity_within_noise_floor(gpu):
    p_base = _vorticity()
    base_layout = generate_bands(p_base.seed, p_base.bands)
    base = _t0(Simulation(p_base, gpu))
    forced = _t0(Simulation(
        _vorticity(belt_fade=0.0, faded_band_index=int(base_layout.fade_index)), gpu))
    np.testing.assert_allclose(forced, base, atol=GPU_NOISE_ATOL)


# ---------------------------------------------------------------- fade effect

def test_belt_fade_full_matches_neighbor_zone_luminance(gpu):
    """belt_fade=1.0: the faded belt's rendered T0 sits at its neighboring
    zones' level -- a pale ghost band, not a dark belt."""
    p = _kinematic(belt_fade=1.0)
    layout = generate_bands(p.seed, p.bands)
    j = int(layout.fade_index)

    base_t0 = _t0(Simulation(_kinematic(), gpu))
    faded_t0 = _t0(Simulation(p, gpu))

    belt_rows = _band_rows(faded_t0, float(layout.edges[j]), float(layout.edges[j + 1]))
    assert belt_rows.any()
    neighbor_means = []
    for k in (j - 1, j + 1):
        if 0 <= k < len(layout.values):
            rows = _band_rows(faded_t0, float(layout.edges[k]), float(layout.edges[k + 1]))
            if rows.any():
                neighbor_means.append(float(faded_t0[rows].mean()))
    assert neighbor_means
    zone_level = float(np.mean(neighbor_means))

    faded_mean = float(faded_t0[belt_rows].mean())
    base_mean = float(base_t0[belt_rows].mean())
    assert abs(faded_mean - zone_level) < 0.05, (
        f"faded belt T0 {faded_mean:.3f} should sit at zone level {zone_level:.3f}"
    )
    assert faded_mean - base_mean > 0.1, "the fade did not lift the belt"

    # Locality: bands outside the fade target and its edge feather are untouched.
    far = _band_rows(faded_t0, float(layout.edges[0]), float(layout.edges[1]))
    if int(layout.fade_index) not in (0, 1) and far.any():
        np.testing.assert_allclose(faded_t0[far], base_t0[far], atol=1e-6)


# ------------------------------------------------------------- outbreak pin

def test_outbreak_pinned_at_35N_erupts_there(gpu):
    """The GWS case: with outbreak_latitude=35 and a mid-run phase, the dev
    run spawns the plume train at ~35N and brightens that band."""
    def params(count: int) -> PlanetParams:
        p = PlanetParams(seed=7)
        p.sim.resolution = 512
        p.sim.dev_steps = 120
        p.storms.outbreak_count = count
        p.storms.outbreak_latitude = 35.0
        p.storms.outbreak_phase = 0.5
        return p

    sim = Simulation(params(1), gpu)
    sim.run_to_completion()
    knots = [v for v in sim.solver.vortices.vortices if v.kind == KIND_OUTBREAK]
    assert knots, "no outbreak vortices alive at the snapshot"
    center = np.deg2rad(35.0)
    for v in knots:
        assert abs(v.lat - center) <= 0.5 * TRAIN_LAT_SPREAD + 1e-6

    base = Simulation(params(0), gpu)
    base.run_to_completion()
    t0_on, t0_off = _t0(sim), _t0(base)
    h = t0_on.shape[0]
    lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    band = np.abs(lats - center) < np.deg2rad(4.0)
    delta = np.abs(t0_on[band] - t0_off[band]).max()
    assert delta > 0.05, f"pinned outbreak left no mark at 35N (max delta {delta:.4f})"
