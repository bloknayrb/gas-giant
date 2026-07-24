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
  * two CROSS-hero sites (detail.comp's heroQ and the calm floor) still take the
    scalar. heroMask SUMS over heroes and is scaled once afterwards, so a
    per-storm form there is a new formulation -- and one that would move the
    flagship preset's shipped output. Left for a visual review; asserted here as
    a KNOWN limit rather than left to be rediscovered.

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


def _synth(gpu, emergences: tuple[float, float] | None) -> np.ndarray:
    """Detail field for the same sim state, overriding each hero's 9th field."""
    p = _params()
    sim = Simulation(p, gpu)
    s = sim.solver
    heroes = hero_centers(sim.vortices, p.storms)
    assert len(heroes) == 2 and len(heroes[0]) == 9, "hero_centers must carry emergence"
    if emergences is not None:
        heroes = [h[:8] + (e,) for h, e in zip(heroes, emergences, strict=True)]
    out = gpu.texture2d((512, 256), 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        p.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, p.detail, heroes=heroes,
        hero_emergence=sim.vortices.scene_emergence(p.storms),
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field.astype(np.float64)


def test_detail_emergence_is_per_storm(gpu):
    """Swapping WHICH hero is emergent changes the detail field — impossible
    while the pass read one scalar for both."""
    a = _synth(gpu, (0.9, 0.1))
    b = _synth(gpu, (0.1, 0.9))
    assert np.abs(a - b).max() > 1e-3


def test_uniform_emergence_matches_the_scalar_path(gpu):
    """The no-op direction: per-hero values all equal to the scene scalar must
    render byte-identically to the legacy 8-tuple path, which supplies no
    per-hero field at all and falls back to that same scalar."""
    legacy = _synth(gpu, None)
    explicit = _synth(gpu, (0.9, 0.9))
    np.testing.assert_array_equal(legacy, explicit)


def test_cross_hero_sites_are_still_scene_wide(gpu):
    """KNOWN LIMIT, pinned so it is not mistaken for per-storm coverage.

    heroQ and the calm floor scale a SUMMED hero mask by the scene scalar, so
    they do not follow a per-hero value. With the fx levers OFF only those two
    sites remain, and a per-hero swap must therefore leave the field EXACTLY
    unchanged. When the cross-hero formulation goes per storm (a visual-review
    change), this test is the one that should start failing."""
    def synth_no_fx(emergences):
        p = _params()
        p.detail.hero_spiral = 0.0
        p.detail.hero_collar_wrap = 0.0
        sim = Simulation(p, gpu)
        s = sim.solver
        heroes = [h[:8] + (e,) for h, e in
                  zip(hero_centers(sim.vortices, p.storms), emergences, strict=True)]
        out = gpu.texture2d((512, 256), 1, "f4", linear=True)
        sim.detail_synth.synthesize(
            p.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
            sim.profile_dyn, out, p.detail, heroes=heroes,
            hero_emergence=sim.vortices.scene_emergence(p.storms),
        )
        field = gpu.read_texture(out)[..., 0]
        out.release()
        return field.astype(np.float64)

    np.testing.assert_array_equal(synth_no_fx((0.9, 0.1)), synth_no_fx((0.1, 0.9)))
