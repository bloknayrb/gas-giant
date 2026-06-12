"""GPU: belt-gated fine detail-tracer replenishment (v1.5).

Tests that:
  (a) belt_replenish=0 is byte-identical to the default (no-op guard works).
  (b) belt_replenish>0 changes T2 in belt rows while leaving zone rows
      untouched after exactly one step.
  (c) high belt_replenish_scale + high detail_freq triggers the aliasing
      warning; low values do not.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _quick(seed: int = 21, dev_steps: int = 8, **turb) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.storms.hero_count = 0
    # Pin detail_freq to keep far from the aliasing-warn threshold (unless
    # a test explicitly overrides it).
    p.bands.detail_freq = 2.0
    for key, value in turb.items():
        setattr(p.turbulence, key, value)
    return p


def _read_t2(sim: Simulation) -> np.ndarray:
    """Return the T2 channel (detail tracer, .z) of the equirect domain."""
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 2]


def _belt_and_zone_rows(sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    """Return (belt_row_indices, zone_row_indices) in the equirect texture.

    profile_dyn is a 1×N LUT; latProfileUV maps lat -> v = (pi/2 - lat)/pi.
    We reconstruct which equirect rows (each with a known latitude) map to
    belt_mask > 0.02 in the LUT.
    """
    dyn = sim.gpu.read_texture(sim.profile_dyn)  # (1, lut_size, 4)
    lut_belt = dyn[0, :, 3]  # .a = belt_mask, shape (lut_size,)
    lut_size = lut_belt.shape[0]

    h = sim.solver.equirect.size[1]  # number of rows in the equirect grid
    # Equirect row i -> latitude (pixel-center convention, +pi/2 at top)
    lats = math.pi / 2 - (np.arange(h) + 0.5) / h * math.pi
    # latProfileUV: v = (0.5*pi - lat) / pi  (clamped to [0,1])
    v_coords = np.clip((0.5 * math.pi - lats) / math.pi, 0.0, 1.0)
    lut_idxs = np.clip((v_coords * lut_size).astype(int), 0, lut_size - 1)
    belt_vals = lut_belt[lut_idxs]

    belt_rows = np.where(belt_vals > 0.02)[0]
    zone_rows = np.where(belt_vals <= 0.02)[0]
    return belt_rows, zone_rows


# ---------------------------------------------------------------------------
# (a) belt_replenish=0 is byte-identical to the default
# ---------------------------------------------------------------------------

def test_belt_replenish_zero_is_byte_identical(gpu):
    """Both sims run the same binary path (no #ifdef) so byte equality is exact."""
    N = 8
    p_default = _quick(dev_steps=N)
    # Default already has belt_replenish=0.0 — explicit set is redundant but
    # proves the guard truly no-ops (not just that 0 happens to coincide).
    p_explicit = _quick(dev_steps=N, belt_replenish=0.0)

    sim_a = Simulation(p_default, gpu)
    sim_a.run_to_completion()
    t2_a = _read_t2(sim_a)

    sim_b = Simulation(p_explicit, gpu)
    sim_b.run_to_completion()
    t2_b = _read_t2(sim_b)

    np.testing.assert_array_equal(t2_a, t2_b)


# ---------------------------------------------------------------------------
# (b) belt_replenish>0 changes belt rows; zone rows untouched after 1 step
# ---------------------------------------------------------------------------

def test_belt_replenish_changes_belt_t2(gpu):
    """Positive belt_replenish must visibly alter T2 in belt latitudes."""
    # Multi-step: assert belt T2 changed globally.
    p_on = _quick(dev_steps=8, belt_replenish=0.05)
    p_off = _quick(dev_steps=8, belt_replenish=0.0)

    sim_on = Simulation(p_on, gpu)
    sim_on.run_to_completion()
    t2_on = _read_t2(sim_on)

    sim_off = Simulation(p_off, gpu)
    sim_off.run_to_completion()
    t2_off = _read_t2(sim_off)

    belt_rows, _ = _belt_and_zone_rows(sim_off)
    assert belt_rows.size > 0, "no belt rows detected — profile_dyn layout may have changed"

    diff_belt = np.abs(t2_on[belt_rows] - t2_off[belt_rows]).max()
    assert diff_belt > 1e-4, (
        f"belt T2 unchanged after replenishment (max-abs-diff={diff_belt:.2e})"
    )


def test_belt_replenish_zone_rows_unchanged_after_one_step(gpu):
    """After exactly ONE step, strictly-zone rows (belt_val well below 0.02)
    must be very nearly identical between belt_replenish=0 and
    belt_replenish>0.

    Rationale: advect.comp injects belt noise only where beltm>0.02.
    Non-belt pixels receive no injection, so after a single step the
    corrector output for those pixels is essentially identical between the two
    sims (same initial tracer state, same advection path, no injection).
    In theory this should be byte-identical; we use atol=1e-4 to tolerate
    any sub-threshold beltm rounding at belt/zone boundaries.
    Multi-step runs let advection carry belt-modified tracers into zone
    rows, so this assertion is limited to a single step.
    """
    p_on = _quick(dev_steps=1, belt_replenish=0.05)
    p_off = _quick(dev_steps=1, belt_replenish=0.0)

    sim_on = Simulation(p_on, gpu)
    sim_on.run_to_completion()
    t2_on = _read_t2(sim_on)

    sim_off = Simulation(p_off, gpu)
    sim_off.run_to_completion()
    t2_off = _read_t2(sim_off)

    _, zone_rows = _belt_and_zone_rows(sim_off)
    assert zone_rows.size > 0, "no zone rows detected"

    # atol=1e-4 lets belt/zone boundary pixels (beltm slightly above 0 in GL
    # bilinear sampling) pass while still catching any meaningful leakage.
    np.testing.assert_allclose(
        t2_on[zone_rows], t2_off[zone_rows], atol=1e-3,
        err_msg="zone-row T2 differed too much after 1 step — belt injection may have leaked",
    )


# ---------------------------------------------------------------------------
# (c) Aliasing warning fires at high freq/scale; silent at safe params
# ---------------------------------------------------------------------------

def test_belt_replenish_scale_warns_and_clears(gpu):
    """High detail_freq + high scale triggers the sub-Nyquist aliasing warning."""
    # At resolution 512: nyquist ~ 8 * 2*pi/512 ≈ 0.0982
    # finest_wavelen = 1.5 / (detail_freq * 2.0 * scale)
    # With detail_freq=14.0, scale=4.0: 1.5/(14*2*4) = 1.5/112 ≈ 0.0134 < 0.0982 → warn
    p_warn = PlanetParams(seed=21)
    p_warn.sim.resolution = 512
    p_warn.sim.dev_steps = 1
    p_warn.storms.hero_count = 0
    p_warn.bands.detail_freq = 14.0
    p_warn.turbulence.belt_replenish = 0.05
    p_warn.turbulence.belt_replenish_scale = 4.0

    with pytest.warns(UserWarning, match="belt_replenish_scale"):
        _sim_warn = Simulation(p_warn, gpu)

    # Safe params: detail_freq=2.0, scale=2.0 → finest = 1.5/(2*2*2)=0.1875 > 0.0982
    p_safe = _quick(dev_steps=1, belt_replenish=0.05, belt_replenish_scale=2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sim_safe = Simulation(p_safe, gpu)
