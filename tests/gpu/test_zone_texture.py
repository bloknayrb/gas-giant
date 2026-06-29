"""GPU tests for detail.zone_texture (fill the detail-starved zones).

Belt interiors get the belt_texture fold + shear-gated filaments; ZONES (low belt
mask, low shear) get neither, so on a banded layout they read as smooth
reduced-detail lanes between the textured belts. zone_texture adds a flow-folded
fold gated by (1-belt) so zones carry their own structure (calmer than belts, not
flat).

Invariants:
  1. zone_texture=0 takes a guarded (un-entered) branch => BYTE-IDENTICAL to not
     setting it, even with another DETAIL_FX lever (mottle) on.
  2. zone_texture>0 adds texture energy to the field (raises std) and changes it.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params(zone_texture: float | None = None, mottle: float = 0.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.detail.mottle = mottle
    if zone_texture is not None:
        p.detail.zone_texture = zone_texture
    return p


def _synth(params: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(params, gpu)
    s = sim.solver
    out = gpu.texture2d((512, 256), 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, params.detail,
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field


# ------------------------------------------------------------- byte-identity

def test_zone_texture_off_byte_identical_with_mottle_on(gpu):
    """zone_texture=0 with mottle on must be byte-identical to mottle alone —
    the guard removes the term from the executed FX program."""
    field_default = _synth(_params(mottle=0.8), gpu)
    field_explicit = _synth(_params(mottle=0.8, zone_texture=0.0), gpu)
    np.testing.assert_array_equal(field_default, field_explicit)


# ------------------------------------------------------------- effect

def test_zone_texture_adds_detail(gpu):
    off = _synth(_params(zone_texture=0.0), gpu)
    on = _synth(_params(zone_texture=1.2), gpu)
    assert np.abs(on - off).max() > 1e-3, "zone_texture produced no visible change"
    assert on.std() > off.std(), (
        f"zone_texture did not add texture energy (off.std={off.std():.4f}, "
        f"on.std={on.std():.4f})"
    )
