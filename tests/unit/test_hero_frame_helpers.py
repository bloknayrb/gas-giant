"""CPU coverage for the hero frame/gate helpers and companion appearance
(PR-43 test review): _hero_bow_gain and the _hero_wake_frame fallbacks feed
premises that two GPU tests depend on but live only in the non-blocking
gpu-full tier; the cast-hero path re-implements the seeded hero's frame logic
and a drift between the two copies would ship silently; companion_brightness
made an explicit byte-identity claim with no test behind it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gasgiant.params.model import CastKind, StormOverride
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import (
    _hero_bow_gain,
    _hero_wake_frame,
    generate_vortices,
)


def _profiles(lat, u=None, t0=None):
    lat = np.asarray(lat, dtype=np.float64)
    return SimpleNamespace(
        lat=lat,
        u=np.asarray(u if u is not None else np.zeros_like(lat)),
        t0_stamp=np.asarray(t0 if t0 is not None else np.zeros_like(lat)),
    )


# ------------------------------------------------------------ _hero_bow_gain

def test_bow_gain_full_on_a_real_band_edge():
    """A t0_stamp step of ~0.2 inside +-1.6 r is a real belt/zone edge ->
    full gain (the smoothstep saturates at 0.14)."""
    lat = np.linspace(-0.5, -0.2, 200)
    t0 = np.where(lat > -0.35, 0.6, 0.4)
    assert _hero_bow_gain(_profiles(lat, t0=t0), -0.35, 0.05) == 1.0


def test_bow_gain_zero_deep_in_a_flat_zone():
    """No boundary within reach -> gain 0 (below the 0.04 banding-noise
    floor): heroBandDeflect must not manufacture a phantom wrap."""
    lat = np.linspace(-0.5, -0.2, 200)
    t0 = np.full_like(lat, 0.5) + 0.01 * np.sin(lat * 40.0)
    assert _hero_bow_gain(_profiles(lat, t0=t0), -0.35, 0.05) < 0.05


def test_bow_gain_empty_window_is_zero():
    """A hero whose +-1.6 r window falls outside the profile grid entirely
    (degenerate placement) must gate OFF, not raise."""
    lat = np.linspace(0.2, 0.5, 100)
    assert _hero_bow_gain(_profiles(lat), -0.35, 0.05) == 0.0


# --------------------------------------------------------- _hero_wake_frame

def test_wake_frame_dead_flow_falls_back_to_legacy():
    """|u| < 0.05 everywhere in the search band -> the legacy authored frame
    (0.5 r equatorward, wake_dir -1): a hero in a stagnant band is a
    plausible user configuration."""
    lat = np.linspace(-0.5, -0.2, 300)
    u = np.full_like(lat, 0.01)
    off, wdir = _hero_wake_frame(_profiles(lat, u=u), -0.35, 0.05)
    assert wdir == -1.0
    assert off == 0.5 * 0.05  # equatorward = +lat for a southern hero


def test_wake_frame_empty_window_falls_back_to_legacy():
    lat = np.linspace(0.2, 0.5, 100)  # search band entirely off-grid
    off, wdir = _hero_wake_frame(_profiles(lat), -0.35, 0.05)
    assert (off, wdir) == (0.5 * 0.05, -1.0)


def test_wake_frame_follows_the_strongest_jet_sign():
    lat = np.linspace(-0.5, -0.2, 300)
    u = np.where((lat > -0.32) & (lat < -0.30), 0.8, 0.02)  # eastward jet
    off, wdir = _hero_wake_frame(_profiles(lat, u=u), -0.35, 0.05)
    assert wdir == 1.0
    assert abs((-0.35 + off) - (-0.31)) < 0.02  # lane sits AT the jet


# -------------------------------------------------- cast-hero frame parity

def test_cast_hero_frame_matches_seeded_hero():
    """The cast path re-implements the emergence wake frame + bow gate; a
    drift between the copies (e.g. dropping the override in one) ships
    silently — this is the cross-site divergence class the GLSL constants
    test guards, on the CPU side. Pin: a cast hero at the SEEDED hero's
    lat/radius carries the identical wake_lat_off / wake_dir / bow_gain."""
    p = load_factory_preset("gas_giant_warm")
    p.storms.hero_longitude = 0.0
    bands = generate_bands(p.seed, p.bands)
    prof = build_profiles(p.seed, bands, p.bands, p.jets)
    seeded = generate_vortices(p.seed, bands, prof, p.storms, p.poles).heroes()[0]

    p2 = load_factory_preset("gas_giant_warm")
    p2.storms.hero_count = 0
    p2.storms.cast = [StormOverride(kind=CastKind.HERO, lat_deg=float(np.degrees(seeded.lat)),
                                lon_deg=0.0, radius=seeded.r_core)]
    reg2 = generate_vortices(p2.seed, bands, prof, p2.storms, p2.poles)
    cast = reg2.heroes()[0]
    assert cast.origin == "cast"
    assert cast.wake_dir == seeded.wake_dir
    assert abs(cast.wake_lat_off - seeded.wake_lat_off) < 1e-12
    assert abs(cast.bow_gain - seeded.bow_gain) < 1e-12


# ------------------------------------------------------ companion_brightness

def test_companion_brightness_reaches_the_companions():
    p = load_factory_preset("gas_giant_warm")
    bands = generate_bands(p.seed, p.bands)
    prof = build_profiles(p.seed, bands, p.bands, p.jets)

    def companions(brightness):
        q = p.model_copy(deep=True)
        q.storms.companion_brightness = brightness
        reg = generate_vortices(q.seed, bands, prof, q.storms, q.poles)
        hero = reg.heroes()[0]
        return [v for v in reg.vortices
                if v.kind != hero.kind and abs(v.lat - hero.lat) < 4 * hero.r_core
                and v.brightness == brightness]

    # warm bakes hero_companions=2: the override must land on exactly them.
    assert len(companions(0.55)) == 2
    assert len(companions(0.71)) == 2


def test_companion_brightness_default_pins_the_pre_lever_constant():
    """The pfield description claims 0.32 reproduces the pre-lever output —
    the model default must stay 0.32 (neptune ships it explicitly)."""
    from gasgiant.params.model import StormsParams

    assert StormsParams.model_fields["companion_brightness"].default == 0.32
