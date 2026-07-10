"""GPU tests for storms.hero_emergence (flow-negotiated hero edge).

The relaxation forcing (advect.comp pass 2) re-imposes the analytic hero stamp
every step, so the flow never owns the storm -> it reads as stamped. hero_emergence
fades the relaxation rate through the hero rim/collar/near-interior (heroRelaxWeight
in vortex_stamp.glsl), so advection folds ambient tracer there instead. The
deep-core anchor keeps full relaxation, and the weight is exactly 1.0 far from any
hero, so everything outside the storm neighborhood is byte-identical.

Invariants:
  1. emergence=0 takes a guarded branch (the call is skipped) => BYTE-IDENTICAL,
     even with other hero levers on.
  2. emergence>0 with NO hero is a no-op: heroRelaxWeight returns exactly 1.0
     everywhere => byte-identical (locality has nothing to bite).
  3. emergence>0 with a hero CHANGES the hero neighborhood (after dev steps) while
     leaving the far field (a hemisphere away) byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(
    emergence: float | None = None,
    hero_count: int = 1,
    dev_steps: int = 60,
    rim_tint: float = 0.0,
    rim_warp: float = 0.0,
) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.storms.hero_count = hero_count
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_rim_tint = rim_tint
    p.storms.hero_rim_warp = rim_warp
    if emergence is not None:
        p.storms.hero_emergence = emergence
    return p


def _developed_tracers(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


# ------------------------------------------------------------- byte-identity

def test_emergence_off_byte_identical_with_other_levers_on(gpu):
    """emergence=0 is a guarded no-op: the heroRelaxWeight call is skipped, so the
    developed tracers are byte-identical even with rim_tint + rim_warp on."""
    base = _developed_tracers(_params(rim_tint=0.7, rim_warp=0.5), gpu)
    same = _developed_tracers(_params(emergence=0.0, rim_tint=0.7, rim_warp=0.5), gpu)
    np.testing.assert_array_equal(base, same)


def test_emergence_no_hero_is_byte_identical(gpu):
    """With NO hero, heroRelaxWeight returns exactly 1.0 everywhere, so rk ==
    u_relax_k (x1.0 is bit-exact) -> emergence>0 is byte-identical to off."""
    off = _developed_tracers(_params(emergence=0.0, hero_count=0), gpu)
    on = _developed_tracers(_params(emergence=1.0, hero_count=0), gpu)
    np.testing.assert_array_equal(off, on)


# ------------------------------------------------------------- effect + locality

def test_emergence_anchors_red_fill_on_hero(gpu):
    """Vorticity mode, emergence on: the hero anchor keeps the prognostic core
    glued to the registry position and the plateau fill paints it red, so the
    developed T3 tint AT the registry position must be strongly warm. (Without
    the anchor the core wanders ~0.2 rad from the stamp and the probe reads
    ~0.0 — the diagnostic that motivated the anchor.)"""
    from gasgiant.params.model import SolverType
    from gasgiant.params.presets import load_factory_preset

    p = load_factory_preset("gas_giant_warm").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 300
    assert p.solver.type == SolverType.VORTICITY
    p.storms.hero_emergence = 1.0
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    tr = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    hero = sim.vortices.heroes()[0]
    h, w = tr.shape[:2]
    row = int((0.5 - hero.lat / np.pi) * h)
    col = int((hero.lon + np.pi) / (2.0 * np.pi) * w)
    # Average T3 over the interior (a few pixels around the center) to be robust
    # to per-pixel mottle; the plateau target is ~hero.tint (0.9).
    patch = tr[row - 2 : row + 3, [c % w for c in range(col - 3, col + 4)], 3]
    assert patch.mean() > 0.3, (
        f"hero interior T3 at the registry position is {patch.mean():.2f} — "
        "the anchored plateau fill did not land on the storm"
    )


def test_emergence_changes_hero_neighborhood_only(gpu):
    """emergence>0 measurably changes the developed tracers near the hero (the
    relaxation there is faded so advection folds the field), while a hemisphere
    away nothing is touched (heroRelaxWeight is exactly 1.0 outside q<3)."""
    off = _developed_tracers(_params(emergence=0.0), gpu)
    on = _developed_tracers(_params(emergence=0.8), gpu)

    # SOME change near the hero (T0 brightness), past the vorticity noise floor.
    delta = np.abs(on[..., 0] - off[..., 0])
    assert delta.max() > 1e-2, "hero_emergence did not change the hero neighborhood"

    # Locality: the far NORTH quarter (hero is at -22.5 deg south, i.e. the
    # southern half) is byte-identical. heroRelaxWeight returns exactly 1.0 there
    # (no hero within q<3), so rk is unchanged and the relaxation math matches
    # bit-for-bit. A real leak would be obvious given the hero is far south.
    h = off.shape[0]
    far = slice(0, h // 4)                       # top quarter = far north
    np.testing.assert_array_equal(on[far], off[far])
