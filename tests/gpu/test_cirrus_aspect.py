"""GPU tests for storms.accent_aspect and storms.companion_aspect.

These elongate the bright ACCENT ovals (KIND_OVAL) and hero COMPANION clouds
(KIND_PEARL) east-west into wispy cirrus streaks, via the same generic
elliptical-q aspect path the hero uses. asp==1.0 (the default) short-circuits
that path, so the default must be BYTE-identical to not specifying it.

The elongated stamp additionally routes through a soft, collar-free glow branch
(no dark rim on a streak); that branch is only reached when asp>1.0, so it too
is byte-identical when off.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _t0(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


def _accent_params(accent_aspect: float | None = None) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.accent_count = 1
    p.storms.accent_latitude = -20.0
    p.storms.accent_radius = 0.06
    p.storms.accent_brightness = 0.3
    if accent_aspect is not None:
        p.storms.accent_aspect = accent_aspect
    return p


def _companion_params(companion_aspect: float | None = None) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    p.storms.hero_companions = 3
    if companion_aspect is not None:
        p.storms.companion_aspect = companion_aspect
    return p


def _lon_lat_extent(diff: np.ndarray, center_lat_deg: float, threshold: float):
    """Longitudinal vs latitudinal extent of the changed region in a +-20 deg
    band around center_lat_deg."""
    h, _w = diff.shape
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo = np.deg2rad(center_lat_deg - 20.0)
    hi = np.deg2rad(center_lat_deg + 20.0)
    band = diff[(lats >= lo) & (lats <= hi), :]
    lon_extent = int(np.sum(band.max(axis=0) > threshold))
    lat_extent = int(np.sum(band.max(axis=1) > threshold))
    return band, lon_extent, lat_extent


# ------------------------------------------------------- accent_aspect

def test_default_accent_aspect_byte_identical(gpu):
    """accent_aspect=1.0 explicit must be BYTE-identical to default params."""
    t0_default = _t0(Simulation(_accent_params(), gpu))
    t0_explicit = _t0(Simulation(_accent_params(accent_aspect=1.0), gpu))
    np.testing.assert_array_equal(t0_default, t0_explicit)


def test_accent_aspect_elongates_streak(gpu):
    """accent_aspect=4.0 spreads the bright accent wider in longitude than latitude."""
    threshold = 1e-4
    t0_1 = _t0(Simulation(_accent_params(accent_aspect=1.0), gpu))
    t0_4 = _t0(Simulation(_accent_params(accent_aspect=4.0), gpu))
    diff = np.abs(t0_4 - t0_1)
    band, lon_extent, lat_extent = _lon_lat_extent(diff, -20.0, threshold)
    assert band.max() > threshold, "accent_aspect=4.0 produced no change — elongation not applied"
    assert lon_extent > 0 and lat_extent > 0
    ratio = lon_extent / max(lat_extent, 1)
    assert ratio > 1.3, (
        f"accent lon/lat extent ratio {ratio:.3f} < 1.3; accent stamp not elongated in longitude"
    )


# ------------------------------------------------------- companion_aspect

def test_default_companion_aspect_byte_identical(gpu):
    """companion_aspect=1.0 explicit must be BYTE-identical to default params."""
    t0_default = _t0(Simulation(_companion_params(), gpu))
    t0_explicit = _t0(Simulation(_companion_params(companion_aspect=1.0), gpu))
    np.testing.assert_array_equal(t0_default, t0_explicit)


def test_companion_aspect_elongates_streaks(gpu):
    """companion_aspect=4.0 elongates the companion clouds east-west."""
    threshold = 1e-4
    t0_1 = _t0(Simulation(_companion_params(companion_aspect=1.0), gpu))
    t0_4 = _t0(Simulation(_companion_params(companion_aspect=4.0), gpu))
    diff = np.abs(t0_4 - t0_1)
    # companions sit within a few core radii of the hero at -22.5 deg
    band, lon_extent, lat_extent = _lon_lat_extent(diff, -22.5, threshold)
    assert band.max() > threshold, "companion_aspect=4.0 produced no change"
    ratio = lon_extent / max(lat_extent, 1)
    assert ratio > 1.3, (
        f"companion lon/lat extent ratio {ratio:.3f} < 1.3; companions not elongated"
    )
