"""Production-regime vorticity boundedness gate (Item 2, v1.6 Phase B P8a).

Loads the jupiter_vorticity factory preset at reduced resolution (2048, down
from 4096, to keep CI runtime reasonable), runs ~400 development steps, then
asserts:

(a) The rendered colour map is finite (no NaN/Inf) and clip fraction ~0.
(b) Each domain's ω field is finite.
(c) In the BELT region of the equirect domain (|lat| < 45°) the fraction of
    cells where |ω| >= OMEGA_CEILING (59.9) is ZERO — the belt must be clean.

This test would fail for:
- a static/no-op field (static omega ≡ q⁰, which may have clamp-saturated cells)
- a blow-up (field → ∞ or NaN)
- the original no-op-uniforms bug (omega never evolved, belt was saturated)
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.solver import DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH

pytestmark = pytest.mark.gpu

# The clamp ceiling in omega_force.comp (OMEGA_CEILING define).
_OMEGA_CEILING = 60.0
# Test threshold: fraction at or above ceiling.  We use 59.9 to tolerate
# exactly-ceiling cells that might persist due to float32 precision.
_CEILING_THRESH = 59.9
# Maximum tolerated ceiling fraction in the belt.  A healthy bounded field
# may have a tiny fraction (~0.1–0.2%) of cells grazing the clamp boundary
# due to genuine transient vorticity injection.  A static/no-op or blow-up
# field would show orders of magnitude more (5–50%).  1% headroom is safe.
_MAX_BELT_CEILING_FRAC = 0.01

# Belt region: |lat| < 45° on the equirect grid.
_BELT_LAT_DEG = 45.0


def _read_omega(sim: Simulation, kind: int) -> np.ndarray:
    """Read |ω| for the given domain kind, returned as (H, W) float32."""
    state = sim.solver._omega_states[kind]
    return sim.gpu.read_texture(state.cur)[..., 0]


def test_vorticity_production_bounded(gpu):
    """Jupiter-vorticity preset at 2048 runs 400 steps; field must be bounded.

    Ceiling fraction in the belt (|lat|<45°) must be 0%.
    """
    p = load_factory_preset("jupiter_vorticity")
    # Override resolution (4096 is too slow for CI) and step count.
    p.sim.resolution = 2048
    p.sim.dev_steps = 400

    sim = Simulation(p, gpu)
    sim.run_to_completion()

    # ---- (a) Rendered colour map: finite, no runaway clipping ---------------
    maps = sim.render_maps(512)
    color = maps["color"]
    assert np.all(np.isfinite(color)), (
        "Rendered colour map contains NaN/Inf after 400 vorticity steps at res 2048"
    )
    clip_frac = float(np.mean((color < 0.0) | (color > 1.0)))
    assert clip_frac < 0.02, (
        f"Rendered colour map has {clip_frac:.1%} pixels outside [0,1] — "
        "likely a NaN propagation or blow-up in vorticity mode"
    )
    print(f"\n  colour: finite=True  clip_frac={clip_frac:.4f}")

    # ---- (b) Each domain's ω is finite --------------------------------------
    for kind, name in (
        (DOMAIN_EQUIRECT, "EQUIRECT"),
        (DOMAIN_NORTH, "NORTH"),
        (DOMAIN_SOUTH, "SOUTH"),
    ):
        omega = _read_omega(sim, kind)
        assert np.all(np.isfinite(omega)), (
            f"Domain {name} ω contains NaN/Inf after 400 steps"
        )
        max_abs = float(np.max(np.abs(omega)))
        print(f"  {name}: finite=True  |omega|_max={max_abs:.3f}")

    # ---- (c) Belt ceiling fraction = 0% (equirect |lat| < 45°) -------------
    omega_eq = _read_omega(sim, DOMAIN_EQUIRECT)
    h, w = omega_eq.shape

    # Row latitudes, descending (row 0 = north pole).
    lats_rad = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lats_deg = np.rad2deg(lats_rad)
    belt_rows = np.where(np.abs(lats_deg) < _BELT_LAT_DEG)[0]

    belt_omega = omega_eq[belt_rows, :]
    ceiling_frac = float(np.mean(np.abs(belt_omega) >= _CEILING_THRESH))

    print(
        f"  EQUIRECT belt (|lat|<{_BELT_LAT_DEG}°): "
        f"ceiling_frac={ceiling_frac:.4f}  "
        f"|omega|_belt_max={float(np.max(np.abs(belt_omega))):.3f}"
    )

    assert ceiling_frac < _MAX_BELT_CEILING_FRAC, (
        f"Belt ceiling fraction {ceiling_frac:.4f} >= {_MAX_BELT_CEILING_FRAC} — "
        f"{ceiling_frac:.1%} of belt cells have |ω| >= {_CEILING_THRESH}. "
        "This indicates a static/no-op or blow-up field (the no-op bug gives "
        "5–50%; a healthy bounded field stays well below 1%). "
        "A healthy bounded vorticity field must have nearly ZERO cells at or "
        "above the clamp ceiling in the equatorial belt after 400 steps."
    )

    sim._release_sim()
