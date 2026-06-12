"""GPU tests for storms.hero_aspect — P3c.

Byte-identity invariant: hero_aspect=1.0 (the default) must produce
EXACTLY the same output as not specifying it, because every elliptical-q
site short-circuits on asp==1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------- helpers

def _params(hero_aspect: float | None = None, hero_spiral: float = 0.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    if hero_aspect is not None:
        p.storms.hero_aspect = hero_aspect
    p.detail.hero_spiral = hero_spiral
    return p


def _t0(sim: Simulation) -> np.ndarray:
    """T0 tracer channel (brightness) from the equirectangular domain."""
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


# ---------------------------------------------------------------- test 1: byte-identical default

def test_default_aspect_round_is_byte_identical(gpu):
    """aspect=1.0 explicit must produce BYTE-IDENTICAL T0 to default params."""
    sim_default = Simulation(_params(), gpu)
    sim_explicit = Simulation(_params(hero_aspect=1.0), gpu)
    t0_default = _t0(sim_default)
    t0_explicit = _t0(sim_explicit)
    np.testing.assert_array_equal(t0_default, t0_explicit)


# ---------------------------------------------------------------- test 2: elongation is visible

def test_aspect_elongates_hero_stamp(gpu):
    """aspect=2.0 must spread the hero signature wider in longitude than latitude.

    Strategy: run aspect=1.0 and aspect=2.0 with the same seed and pinned
    latitude.  Compute |diff| between the two T0 fields.  In the region
    around the hero latitude, measure the longitudinal extent (number of
    non-zero columns) and the latitudinal extent (number of non-zero rows)
    of the |diff| region.  For aspect=2.0 the lon-extent must meaningfully
    exceed lat-extent; for aspect=1.0 they should be roughly equal.
    """
    threshold = 1e-4  # float32 precision floor

    sim_1 = Simulation(_params(hero_aspect=1.0), gpu)
    sim_2 = Simulation(_params(hero_aspect=2.0), gpu)

    t0_1 = _t0(sim_1)
    t0_2 = _t0(sim_2)

    diff = np.abs(t0_2 - t0_1)

    # Restrict to the hero latitude band (+-20 deg around -22.5 deg)
    h, w = diff.shape
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lat_lo = np.deg2rad(-42.5)
    lat_hi = np.deg2rad(-2.5)
    band_rows = (lats >= lat_lo) & (lats <= lat_hi)
    band = diff[band_rows, :]

    assert band.max() > threshold, (
        "aspect=2.0 produced no change near the hero — elongation not applied"
    )

    # Longitudinal extent: count columns with any value above threshold.
    lon_extent = int(np.sum(band.max(axis=0) > threshold))
    # Latitudinal extent: count rows in the band with any value above threshold.
    lat_extent = int(np.sum(band.max(axis=1) > threshold))

    assert lon_extent > 0 and lat_extent > 0, (
        f"Expected non-zero extents; got lon={lon_extent}, lat={lat_extent}"
    )
    ratio = lon_extent / max(lat_extent, 1)
    assert ratio > 1.3, (
        f"Longitudinal/latitudinal extent ratio is {ratio:.3f} < 1.3; "
        "hero stamp does not appear elongated in longitude"
    )

    # Sanity: aspect=1.0 self-diff is zero (same binary).
    sim_1b = Simulation(_params(hero_aspect=1.0), gpu)
    t0_1b = _t0(sim_1b)
    np.testing.assert_array_equal(t0_1, t0_1b)

    # aspect=1.0 vs 2.0 extents: the round case should have ratio closer to 1.
    # (We just assert the elongated case is more asymmetric than 1.3.)


# ---------------------------------------------------------------- test 3: detail byte-identical

def test_detail_default_aspect_byte_identical(gpu):
    """DetailSynth heroMask + spiral (hero_spiral=0.7) must be byte-identical
    at aspect=1.0 explicit vs default (short-circuit path exercised)."""
    from gasgiant.engine.snapshot import hero_centers

    def _synth(params: PlanetParams) -> np.ndarray:
        sim = Simulation(params, gpu)
        s = sim.solver
        out = gpu.texture2d((512, 256), 1, "f4", linear=True)
        sim.detail_synth.synthesize(
            params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
            sim.profile_dyn, out, params.detail,
            heroes=hero_centers(sim.vortices),
        )
        field = gpu.read_texture(out)[..., 0]
        out.release()
        return field

    p_default = _params(hero_spiral=0.7)
    p_explicit = _params(hero_aspect=1.0, hero_spiral=0.7)

    field_default = _synth(p_default)
    field_explicit = _synth(p_explicit)

    np.testing.assert_array_equal(field_default, field_explicit)
