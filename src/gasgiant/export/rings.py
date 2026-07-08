"""Saturn-style ring texture strip (T16).

Rings are a Blender-only product feature: a 2048x64 RGBA radial strip written
as a SEPARATE exported map (``rings.exr``), rebuilt as an annulus by the
importer. They are NOT part of the equirect map set and are invisible in the
GUI preview, so enabling them never touches the color/height/emission render
path (the p05 render hash is unaffected).

The radial brightness/coverage profile is built from a BOUNDED, hardcoded
optical-depth table -- NOT open-ended seeded look-dev. The control points model
the real gross structure of Saturn's rings as seen edge-on / in reflected light:

    Ring     radius (km, from center)   normal optical depth (approx)
    -------  -------------------------   ----------------------------
    C ring    74,500 -  92,000           ~0.05 - 0.15  (faint, dusky)
    B ring    92,000 - 117,580           ~0.4  - 2.5   (dense, brightest)
    Cassini  117,580 - 122,170           ~0.05 - 0.15  (the famous GAP)
    A ring   122,170 - 136,780           ~0.4  - 1.0
      Encke  ~133,570                     narrow (~325 km) empty gap in A

Source structure: Cuzzi et al. / Voyager & Cassini radio/stellar occultation
optical-depth profiles (the canonical values quoted in planetary-science texts).
Radii are expressed here as a FRACTION of the exported span
(physical.ring_inner_km .. ring_outer_km) so the table is independent of the
chosen extent; optical depth tau is turned into alpha coverage via Beer-Lambert
(alpha = 1 - exp(-tau)), and reflectance brightness is a saturating function of
tau. Seeded fine-grain ringlet variation (subseed(seed, "rings")) is layered on
top. Every output value is deterministic and bounded to [0, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gasgiant.params.seeds import subseed

# Radial optical-depth control points: (radius_fraction in [0,1], normal tau).
# fraction 0 = ring_inner_km, fraction 1 = ring_outer_km. Kept intentionally
# small and hand-authored -- this is a physical structure table, not noise.
_OPTICAL_DEPTH_TABLE: tuple[tuple[float, float], ...] = (
    (0.000, 0.00),   # inner C edge: fade in from transparent
    (0.020, 0.08),   # C ring begins (faint)
    (0.250, 0.12),   # C ring body
    (0.281, 0.90),   # B ring inner: sharp rise
    (0.360, 1.80),   # B ring dense inner
    (0.500, 2.20),   # B ring peak (densest, brightest)
    (0.620, 1.70),   # B ring outer body
    (0.690, 1.30),   # B ring outer edge
    (0.696, 0.10),   # Cassini division: sharp drop into the gap
    (0.730, 0.08),   # Cassini division floor (the GAP)
    (0.762, 0.12),   # Cassini division outer
    (0.766, 0.60),   # A ring inner: rise
    (0.850, 0.90),   # A ring body
    (0.945, 0.80),   # A ring outer body
    (0.949, 0.10),   # Encke gap (narrow dip in A)
    (0.956, 0.80),   # A ring resumes past Encke
    (0.990, 0.50),   # A ring outer taper
    (1.000, 0.00),   # outer edge: fade to transparent
)

RING_WIDTH = 2048   # radial samples (inner -> outer)
RING_HEIGHT = 64    # tangential rows


def _radial_optical_depth() -> np.ndarray:
    """Piecewise-linear interpolation of the control-point table onto RING_WIDTH
    radial samples. Returns a (RING_WIDTH,) float32 array of normal optical depth
    (>= 0, unbounded above -- the B ring peaks past 2)."""
    fracs = np.array([p[0] for p in _OPTICAL_DEPTH_TABLE], dtype=np.float64)
    taus = np.array([p[1] for p in _OPTICAL_DEPTH_TABLE], dtype=np.float64)
    # Sample the CENTER of each radial texel so the strip is symmetric.
    x = (np.arange(RING_WIDTH, dtype=np.float64) + 0.5) / RING_WIDTH
    return np.interp(x, fracs, taus).astype(np.float32)


def ring_strip(params: Any) -> np.ndarray:
    """Build the (RING_WIDTH, RING_HEIGHT, 4) RGBA ring strip for ``params``
    (a PlanetParams). Axis 0 is radial (inner -> outer), axis 1 is tangential.

    Alpha is Beer-Lambert coverage from the bounded optical-depth table
    (scaled by rings.opacity); RGB is a warm ice tint scaled by a saturating
    reflectance of optical depth (times rings.brightness). Seeded fine grain
    (subseed(seed, "rings")) adds bounded ringlet variation. All values are
    deterministic and clamped to [0, 1]."""
    rings = params.rings
    tau = _radial_optical_depth()  # (W,), >= 0

    rng = subseed(params.seed, "rings")
    grain_amt = float(rings.fine_grain)
    # Fine grain: per-radial multiplicative ringlet variation (bounded), plus a
    # faint per-row tangential wobble so the 64 rows are not a dead copy. Both
    # are gated by fine_grain and by tau (empty gaps stay empty).
    radial_grain = 1.0 + grain_amt * 0.6 * (rng.random(RING_WIDTH).astype(np.float32) - 0.5)
    row_grain = 1.0 + grain_amt * 0.15 * (rng.random(RING_HEIGHT).astype(np.float32) - 0.5)
    # (W, H) optical depth after grain; keep it non-negative.
    tau_grid = np.clip(
        tau[:, None] * radial_grain[:, None] * row_grain[None, :], 0.0, None
    )

    # Beer-Lambert coverage; opacity scales it, then clamp.
    alpha = np.clip((1.0 - np.exp(-tau_grid)) * float(rings.opacity), 0.0, 1.0)

    # Reflectance: a saturating function of optical depth (denser rings reflect
    # more, but roll off). Bounded to [0, 1] before the tint/brightness apply.
    reflect = np.clip(1.0 - np.exp(-1.3 * tau_grid), 0.0, 1.0)
    tint = np.asarray(rings.tint_color, dtype=np.float32)  # (3,)
    brightness = float(rings.brightness)
    rgb = np.clip(
        reflect[..., None] * tint[None, None, :] * brightness, 0.0, 1.0
    ).astype(np.float32)

    strip = np.empty((RING_WIDTH, RING_HEIGHT, 4), dtype=np.float32)
    strip[..., :3] = rgb
    strip[..., 3] = alpha
    return np.ascontiguousarray(strip)
