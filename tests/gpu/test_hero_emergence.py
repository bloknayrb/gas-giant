"""GPU tests for storms.hero_emergence (flow-negotiated hero edge).

The relaxation forcing (advect.comp pass 2) re-imposes the analytic hero stamp
every step, so the flow never owns the storm -> it reads as stamped. hero_emergence
fades the relaxation rate through the hero rim/collar/near-interior (heroRelaxWeight
in vortex_stamp.glsl), so advection folds ambient tracer there instead. The
deep-core anchor keeps full relaxation, and the weight is exactly 1.0 far from any
hero, so everything outside the storm neighborhood is byte-identical.

The whole feature compiles as a HERO_EMERGENCE preprocessor variant selected by
solver._domain_defines (emergence > 0 AND a hero exists), so "off is the
pre-feature program" is structural — pinned by the kinematic source hashes
(tests/unit/test_kinematic_kernels_pinned.py) and the p05 render-hash gate, not
re-provable at runtime. What CAN be pinned at runtime, and is below:
  1. the default kinematic path (emergence=0, rim levers on) is deterministic
     across full Simulation rebuilds;
  2. emergence>0 with NO hero selects the default program (predicate pin);
  3. with the variant COMPILED (hero present, emergence>0), the far field is
     byte-identical — the runtime forced-variant no-op, hero-locality edition;
  4. the anchored plateau fill actually lands on the registry position.

Every byte-exact assert here relies on the KINEMATIC solver path (the vorticity
SOR solve carries a documented ~1e-3 noise floor and is never byte-compared);
_params asserts the mode so a future default-solver flip fails loudly instead
of flaking against the noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(
    emergence: float = 0.0,
    hero_count: int = 1,
    rim_tint: float = 0.0,
    rim_warp: float = 0.0,
) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    p.storms.hero_count = hero_count
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_rim_tint = rim_tint
    p.storms.hero_rim_warp = rim_warp
    p.storms.hero_emergence = emergence
    # The byte-exact asserts in this file are only valid on the kinematic path
    # (vorticity output is tolerance-compared everywhere else in the suite).
    assert p.solver.type == SolverType.KINEMATIC
    return p


def _developed_tracers(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


# ------------------------------------------------------------- byte-identity

def test_emergence_off_byte_identical_with_other_levers_on(gpu):
    """Determinism canary for the emergence=0 program: both runs use IDENTICAL
    params (emergence defaults to 0.0 in _params — this is deliberate, not an
    off-vs-on comparison), so this pins that the default kinematic path with
    rim_tint + rim_warp on is reproducible across two full Simulation builds.
    The actual off == pre-feature guarantee is structural (variant not
    compiled) and is pinned by the source hashes + p05, not runtime-testable."""
    base = _developed_tracers(_params(rim_tint=0.7, rim_warp=0.5), gpu)
    same = _developed_tracers(_params(emergence=0.0, rim_tint=0.7, rim_warp=0.5), gpu)
    np.testing.assert_array_equal(base, same)


def test_emergence_no_hero_is_byte_identical(gpu):
    """With NO hero, _domain_defines does not select the HERO_EMERGENCE variant
    (the predicate requires a hero), so emergence>0 runs the DEFAULT program —
    byte-identical to off by construction. This pins the predicate: a no-hero
    config must never pay the variant's per-pixel vortex-SSBO scan for a
    guaranteed no-op (heroRelaxWeight would return exactly 1.0 everywhere)."""
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
    """The runtime forced-variant no-op test (CLAUDE.md lever rule), hero-local
    edition: with a hero present and emergence>0 the HERO_EMERGENCE variant IS
    compiled, and it must (a) measurably change the developed tracers near the
    hero (the relaxation there is faded so advection folds the field) while
    (b) leaving the far field byte-identical — heroRelaxWeight returns exactly
    1.0 outside q<3.6, so rk == u_relax_k bit-for-bit out there."""
    off = _developed_tracers(_params(emergence=0.0), gpu)
    on = _developed_tracers(_params(emergence=0.8), gpu)

    # SOME change near the hero (T0 brightness), past the vorticity noise floor.
    delta = np.abs(on[..., 0] - off[..., 0])
    assert delta.max() > 1e-2, "hero_emergence did not change the hero neighborhood"

    # Locality: the far NORTH quarter (hero is at -22.5 deg south, i.e. the
    # southern half) is byte-identical. heroRelaxWeight returns exactly 1.0 there
    # (no hero within q<3.6), so rk is unchanged and the relaxation math matches
    # bit-for-bit. A real leak would be obvious given the hero is far south.
    h = off.shape[0]
    far = slice(0, h // 4)                       # top quarter = far north
    np.testing.assert_array_equal(on[far], off[far])
