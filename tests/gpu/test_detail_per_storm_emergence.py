"""M2-C: per-storm hero emergence on the RENDER detail pass.

The sim side has been per-storm since M2-B, but detail.comp read one scalar
`u_hero_emergence`, so two placed heroes at different emergence got the same
detail treatment. Its read sites split cleanly in two:

  * three sites INSIDE the per-hero loop (spiral pitch, spiral window, collar
    window) now index `u_hero_emergence_arr[i]`. For a uniform scene that is an
    operand substitution with an equal value -> byte-identical, gated by the
    `render_*` configs in scripts/m2b_emergence_hash.py (which drive the fx
    levers that own those sites, else DETAIL_FX does not compile and they are
    not in the program at all).
  * two CROSS-hero sites (detail.comp's heroQ and the calm floor) follow via
    heroMaskFaded/heroCalmFloor. heroMask SUMS over heroes and was scaled once
    afterwards, so this is a NEW formulation, not a substitution -- it cannot be
    bit-identical in general. It IS exact for a single hero (one summed term,
    clamp is the identity), which is every factory preset, so nothing re-bakes.

These call synthesize() directly with hand-built hero tuples so the comparison
isolates the DETAIL pass: driving emergence through params would also move the
sim, and the tracer difference would swamp the render difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.snapshot import hero_centers
from gasgiant.params.model import CastKind, PlanetParams, StormOverride

pytestmark = pytest.mark.gpu

_PAIR = ((-20.0, -7.0), (-20.0, 7.0))


def _params() -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 0
    p.storms.hero_emergence = 0.9
    p.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=lat, lon_deg=lon, radius=0.1)
        for lat, lon in _PAIR
    ]
    # The three per-hero sites live in the DETAIL_FX-only spiral/collar block.
    p.detail.intensity = 0.6
    p.detail.hero_spiral = 0.8
    p.detail.hero_collar_wrap = 0.7
    return p


def _synth(gpu, emergences: tuple[float, float] | None, *, fx: bool = True) -> np.ndarray:
    """Detail field for the same sim state. ``emergences`` overrides each hero's
    9th field; None TRUNCATES to the legacy 8-tuple — no per-hero field at all,
    which is the scalar-fallback path every non-facade caller takes and the one
    the no-op test has to actually exercise.

    ``fx=False`` zeroes the two levers that own the per-hero read sites. Every
    other fx pfield defaults to 0.0, so DETAIL_FX then does not compile at all
    and ONLY the cross-hero, scene-wide sites remain live."""
    p = _params()
    if not fx:
        p.detail.hero_spiral = 0.0
        p.detail.hero_collar_wrap = 0.0
    sim = Simulation(p, gpu)
    heroes = hero_centers(sim.vortices, p.storms)
    assert len(heroes) == 2 and len(heroes[0]) == 9, "hero_centers must carry emergence"
    if emergences is None:
        heroes = [h[:8] for h in heroes]
    else:
        heroes = [h[:8] + (e,) for h, e in zip(heroes, emergences, strict=True)]
    return _render(gpu, sim, p, heroes)


def _render(gpu, sim, params: PlanetParams,
            heroes: list[tuple[float, ...]]) -> np.ndarray:
    """Run ONLY the detail pass over an already-developed sim, for a hand-built
    hero list. Isolating the pass is the whole point of this file: driving
    emergence through params would also move the sim, and the tracer difference
    would swamp the render difference."""
    s = sim.solver
    out = gpu.texture2d((512, 256), 1, "f4", linear=True)
    try:
        sim.detail_synth.synthesize(
            params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
            sim.profile_dyn, out, params.detail, heroes=heroes,
            hero_emergence=sim.vortices.scene_emergence(params.storms),
        )
        return gpu.read_texture(out)[..., 0].astype(np.float64)
    finally:
        out.release()


def test_detail_emergence_is_per_storm(gpu):
    """Swapping WHICH hero is emergent changes the detail field — impossible
    while the pass read one scalar for both."""
    a = _synth(gpu, (0.9, 0.1))
    b = _synth(gpu, (0.1, 0.9))
    assert np.abs(a - b).max() > 1e-3


def test_uniform_emergence_matches_the_scalar_path(gpu):
    """The no-op direction: per-hero values all equal to the scene scalar must
    render byte-identically to the legacy 8-tuple path, which supplies no
    per-hero field at all and falls back to that same scalar. This is the
    guarantee every existing hero_centers(registry) caller rides on."""
    legacy = _synth(gpu, None)
    explicit = _synth(gpu, (0.9, 0.9))
    np.testing.assert_array_equal(legacy, explicit)


def test_cross_hero_sites_are_per_storm_too(gpu):
    """The CROSS-hero sites (heroQ, the serene-moat calm floor) follow per
    storm as well, via heroMaskFaded/heroCalmFloor.

    Turning the fx levers OFF strips DETAIL_FX entirely, so the three per-hero
    loop sites are not in the program at all and ONLY the cross-hero pair can
    account for a difference. This test previously pinned the opposite — that
    those two sites were scene-wide — and was written to start failing exactly
    here."""
    a = _synth(gpu, (0.9, 0.1), fx=False)
    b = _synth(gpu, (0.1, 0.9), fx=False)
    assert np.abs(a - b).max() > 1e-3


def test_single_hero_fallback_matches_the_scene_scalar(gpu):
    """The 8-tuple fallback again at hero_count=1 — the arity every factory
    preset actually ships.

    SCOPE, because an earlier version of this test claimed more than it can
    deliver: it does NOT compare the new cross-hero formulation against the old
    scalar one. It cannot. The old formulation no longer exists in the process
    (detail.comp's scalar u_hero_emergence was deleted once its last two reads
    moved to the array), so both sides here run the SAME program, and the CPU
    packs byte-identical uniform data for each — np.full(3, scene_emergence)
    for the 8-tuple, an explicit 0.9 into slot 0 for the 9-tuple. Identical
    float32 in, identical pixels out, whatever the shader does.

    The single-hero exactness claim — that all six presets keep their shipped
    output — is a CROSS-TREE claim and only a cross-tree measurement can carry
    it: scripts/m2b_emergence_hash.py's render_bare/render_shape_taper, checked
    out against the pre-change sources (measured 7/8 identical, render_two_heroes
    the only mover). This test guards the narrower thing it can see: that the
    fallback fill still resolves correctly when only one hero is present."""
    p = _params()
    p.storms.cast = p.storms.cast[:1]          # a single emergent hero
    sim = Simulation(p, gpu)
    heroes = hero_centers(sim.vortices, p.storms)
    assert len(heroes) == 1

    # Legacy 8-tuple (scene scalar broadcast into every slot) vs the explicit
    # per-hero value carrying that same scalar: byte-identical, or the fallback
    # fill has stopped resolving.
    np.testing.assert_array_equal(
        _render(gpu, sim, p, [heroes[0][:8]]),
        _render(gpu, sim, p, [heroes[0][:8] + (0.9,)]),
    )
