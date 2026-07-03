"""GPU tests for the W5 per-storm color levers (review Top-10 #5).

  - Default no-op: every new lever at its default (plus the forced-variant
    stamp_tint_contrast == stamp_contrast) renders within the GPU noise floor
    on the vorticity presets and BYTE-IDENTICAL on a kinematic preset.
  - hero_brightness < 0 renders a dark storm (Neptune GDS, A06/B5-1).
  - Accent ovals survive a real dev run as a coherent red oval at -33 deg
    (A01 Oval BA — small Gaussian ovals dissipate, F07, hence the post-dev
    coherence gate + oval_solid_core pairing).
  - hero_companions renders bright clouds pinned beside the hero (B5-5).

Tolerance policy per CLAUDE.md: vorticity-path comparisons use the 1e-2 GPU
noise floor; the kinematic path asserts byte-exact.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.vortices import KIND_HERO, KIND_PEARL

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2


def _shrunk(preset: str, seed: int = 7, steps: int = 60, res: int = 512):
    p = load_factory_preset(preset).model_copy(update={"seed": seed})
    p.sim.resolution = res
    p.sim.dev_steps = steps
    return p


def _force_defaults(p):
    """Set every W5 lever to its default-equivalent forced variant: values that
    MUST reproduce the pre-lever output exactly (the lever-author checklist's
    forced-variant no-op)."""
    p.storms.hero_tint = 0.9
    p.storms.hero_brightness = 0.05
    p.storms.hero_companions = 0
    p.storms.accent_count = 0
    # Appearance params are inert at count=0.
    p.storms.accent_latitude = -33.0
    p.storms.accent_tint = 0.77
    p.storms.accent_brightness = 0.3
    p.storms.accent_radius = 0.06
    # Explicit follow value == legacy coupled arithmetic.
    p.storms.stamp_tint_contrast = p.storms.stamp_contrast
    return p


def _render(p, gpu, width: int = 512) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(width)["color"][..., :3].astype(np.float64)
    finally:
        sim._release_sim()


def _render_and_registry(p, gpu, width: int = 512):
    sim = Simulation(p, gpu)
    try:
        color = sim.render_maps(width)["color"][..., :3].astype(np.float64)
        return color, list(sim.vortices.vortices)
    finally:
        sim._release_sim()


def _pix(lat: float, lon: float, h: int, w: int) -> tuple[int, int]:
    x = int((lon + np.pi) / (2 * np.pi) * w) % w
    y = int(np.clip((np.pi / 2 - lat) / np.pi * h, 0, h - 1))
    return y, x


def _disc_mean(img: np.ndarray, lat: float, lon: float, rad: float) -> np.ndarray:
    """Mean color over the pixels within great-circle-ish radius rad (radians)
    of (lat, lon) — small-patch equirect approximation."""
    h, w = img.shape[:2]
    y0, x0 = _pix(lat, lon, h, w)
    ry = max(int(rad / np.pi * h), 1)
    rx = max(int(rad / (2 * np.pi) * w / max(np.cos(lat), 0.2)), 1)
    ys = np.arange(y0 - ry, y0 + ry + 1)
    xs = np.arange(x0 - rx, x0 + rx + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = ((yy - y0) / ry) ** 2 + ((xx - x0) / rx) ** 2 <= 1.0
    patch = img[np.clip(yy, 0, h - 1), xx % w]
    return patch[mask].mean(axis=0)


# ------------------------------------------------------------- default no-op

@pytest.mark.parametrize("preset", ["gas_giant_warm", "jupiter_vorticity"])
def test_forced_defaults_noop_vorticity(gpu, preset):
    """Vorticity presets: forced-default variant within the GPU noise floor."""
    base = _render(_shrunk(preset), gpu)
    forced = _render(_force_defaults(_shrunk(preset)), gpu)
    assert np.abs(forced - base).max() < GPU_NOISE_ATOL, (
        f"W5 levers at forced defaults perturbed {preset} beyond the noise floor"
    )


def test_forced_defaults_byte_identical_kinematic(gpu):
    """Kinematic path: forced defaults must be BYTE-identical (policy: never
    loosen a kinematic identity to a tolerance)."""
    base = _render(_shrunk("jupiter_like"), gpu)
    forced = _render(_force_defaults(_shrunk("jupiter_like")), gpu)
    np.testing.assert_array_equal(forced, base)


# ------------------------------------------------------------- dark hero A06

def test_dark_hero_on_ice_giant(gpu):
    """hero_brightness=-0.4 + hero_tint=-0.8 renders the hero region measurably
    darker than its band surroundings (the Neptune GDS: negative brightness
    darkens T0; negative tint pulls toward the storm LUT's dark-navy cool end)."""
    p = _shrunk("ice_giant", steps=60)
    p.storms.hero_latitude = -30.0
    p.storms.hero_brightness = -0.4
    p.storms.hero_tint = -0.8
    color, reg = _render_and_registry(p, gpu)
    hero = [v for v in reg if v.kind == KIND_HERO][0]
    inside = _disc_mean(color, hero.lat, hero.lon, 0.5 * hero.r_core).mean()
    # Same-latitude ring far from the storm (and its wake, which trails west).
    far1 = _disc_mean(color, hero.lat,
                      (hero.lon + np.pi * 0.5) % (2 * np.pi) - np.pi,
                      2 * hero.r_core).mean()
    far2 = _disc_mean(color, hero.lat,
                      (hero.lon - np.pi * 0.5 + np.pi) % (2 * np.pi) - np.pi,
                      2 * hero.r_core).mean()
    surround = 0.5 * (far1 + far2)
    assert inside < surround - 0.05, (
        f"dark hero not darker than surroundings: inside={inside:.3f} "
        f"surround={surround:.3f}"
    )


# --------------------------------------------------- accent post-dev (A01/F07)

def test_accent_oval_survives_dev_run_coherent(gpu):
    """POST-DEV coherence: a red accent oval at -33 deg must survive a real
    vorticity dev run as a compact red spot (small Gaussian ovals dissipate —
    F07 — so this pairs accent_radius 0.06 with oval_solid_core)."""
    p = _shrunk("jupiter_vorticity", steps=300)
    p.storms.hero_count = 0          # keep the red hero out of the -33 band
    p.storms.accent_count = 1
    p.storms.accent_latitude = -33.0
    p.storms.accent_tint = 0.9
    p.storms.accent_brightness = 0.25
    p.storms.accent_radius = 0.06
    p.storms.oval_solid_core = 1.0
    color, reg = _render_and_registry(p, gpu)
    accents = [v for v in reg if v.tint == 0.9 and v.brightness == 0.25]
    assert len(accents) == 1
    acc = accents[0]  # drifted final registry position

    # The tracer blob wanders a few degrees off the registry position over a
    # long dev run (it rides the full local flow; the registry drifts with the
    # ambient jet only) — so find the red PEAK inside a local window around the
    # registry position, then assert the blob is strong and compact there.
    h, w = color.shape[:2]
    redness = color[..., 0] - 0.5 * (color[..., 1] + color[..., 2])
    y0, x0 = _pix(acc.lat, acc.lon, h, w)
    dy = int(np.deg2rad(8.0) / np.pi * h)
    dx = int(np.deg2rad(20.0) / (2 * np.pi) * w)
    win = redness[np.clip(np.arange(y0 - dy, y0 + dy + 1), 0, h - 1)][
        :, np.arange(x0 - dx, x0 + dx + 1) % w]
    peak = float(win.max())
    background = float(np.median(win))
    assert peak > background + 0.12, (
        f"accent oval did not survive the dev run as a red spot: "
        f"peak={peak:.4f} background={background:.4f}"
    )
    # Coherence: the surviving red is one compact spot (rms spread well under
    # the core radius), not a sheared-out streak or scattered eddies.
    ys, xs = np.nonzero(win > 0.5 * peak)
    lat_rms = float(ys.std()) * np.pi / h
    lon_rms = float(xs.std()) * (2 * np.pi / w) * np.cos(acc.lat)
    assert lat_rms < 0.05 and lon_rms < 0.05, (
        f"accent red is smeared, not compact: lat_rms={lat_rms:.4f} "
        f"lon_rms={lon_rms:.4f} rad (r_core={acc.r_core})"
    )


def test_accent_flip_a01_two_color_states(gpu):
    """A01 flip: the same preset renders white ovals AND a red accent oval —
    the two-color-state epoch. Verified as accent-on vs accent-off: at the
    accent site the on-render carries a red oval the base render lacks.

    NOTE deliberately no far-field locality assertion: in vorticity mode a new
    vortex perturbs the global SOR solve and the dev run is chaotic, so remote
    filaments legitimately shift (registry-level isolation of the base
    population is asserted in the unit tests instead)."""
    base_p = _shrunk("jupiter_vorticity", steps=60)
    on_p = _shrunk("jupiter_vorticity", steps=60)
    on_p.storms.accent_count = 1
    on_p.storms.accent_latitude = -33.0
    on_p.storms.accent_tint = 0.9
    on_p.storms.accent_brightness = 0.25
    on_p.storms.accent_radius = 0.06
    base = _render(base_p, gpu)
    on, reg = _render_and_registry(on_p, gpu)
    acc = [v for v in reg if v.tint == 0.9 and v.brightness == 0.25][0]

    diff = on - base
    assert np.abs(diff).max() > GPU_NOISE_ATOL, "accent oval did not render"
    # Red push at the accent site: R rises more than G/B.
    d = _disc_mean(diff, acc.lat, acc.lon, 1.2 * acc.r_core)
    assert d[0] > d[1] and d[0] > d[2], f"accent diff is not a red push: {d}"
    # Two color states: the accent site is red with the lever on, and was not
    # red in the base render (the ovals there are white).
    h, w = on.shape[:2]
    y0, x0 = _pix(acc.lat, acc.lon, h, w)
    dy = int(np.deg2rad(6.0) / np.pi * h)
    dx = int(np.deg2rad(12.0) / (2 * np.pi) * w)
    rows = np.clip(np.arange(y0 - dy, y0 + dy + 1), 0, h - 1)
    cols = np.arange(x0 - dx, x0 + dx + 1) % w
    red_on = (on[..., 0] - 0.5 * (on[..., 1] + on[..., 2]))[rows][:, cols].max()
    red_base = (base[..., 0] - 0.5 * (base[..., 1] + base[..., 2]))[rows][:, cols].max()
    assert red_on > red_base + 0.1, (
        f"no red/white flip at the accent site: on={red_on:.3f} base={red_base:.3f}"
    )


# ------------------------------------------------------------ companions B5-5

def test_hero_companions_render_bright_beside_hero(gpu):
    """hero_companions=2 adds bright clouds near the hero and nothing else."""
    base_p = _shrunk("ice_giant", steps=0)
    on_p = _shrunk("ice_giant", steps=0)
    for q in (base_p, on_p):
        q.storms.hero_latitude = -30.0
    on_p.storms.hero_companions = 2
    base = _render(base_p, gpu)
    on, reg = _render_and_registry(on_p, gpu)
    comps = [v for v in reg if v.kind == KIND_PEARL]
    assert len(comps) == 2
    diff = on - base
    assert np.abs(diff).max() > GPU_NOISE_ATOL, "companions did not render"
    for c in comps:
        d = _disc_mean(diff, c.lat, c.lon, c.r_core)
        assert d.mean() > 0.01, f"companion at ({c.lat:.2f},{c.lon:.2f}) not bright"
