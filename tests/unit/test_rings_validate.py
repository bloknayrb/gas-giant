"""T16: `gasgiant validate` on a rings-carrying map set (no GL).

The rings map is a RADIAL optical-depth strip, not a sphere map: ringlet/gap
edges are tangentially-uniform jumps — exactly the signature the sphere seam
checks (wrap/pole/latitude-band continuity) flag. validate_mapset must give
the rings entry the finiteness check ONLY, so a legal high-contrast strip
(e.g. rings.fine_grain at its bound) can never spuriously FAIL validation.
"""

from __future__ import annotations

import numpy as np

from gasgiant.export.manifest import build_manifest, write_manifest
from gasgiant.export.writers import write_exr_gray, write_exr_rgba, write_png16_rgb_u16
from gasgiant.validate.seams import validate_mapset


def _write_mapset(tmp_path, rings: np.ndarray):
    w, h = 64, 32
    color = np.full((h, w, 3), 32768, dtype=np.uint16)
    height = np.full((h, w), 0.5, dtype=np.float32)
    write_png16_rgb_u16(tmp_path / "color.png", color, 2)
    write_exr_gray(tmp_path / "height.exr", height)
    write_exr_rgba(tmp_path / "rings.exr", rings)
    maps = {
        "color": {"file": "color.png", "format": "png16",
                  "colorspace": "srgb", "channels": 3},
        "height": {"file": "height.exr", "format": "exr32f",
                   "colorspace": "non-color", "channels": 1},
        "rings": {"file": "rings.exr", "format": "exr32f",
                  "colorspace": "non-color", "channels": 4,
                  "convention": "radial_inner_to_outer_alpha_coverage"},
    }
    manifest = build_manifest(
        name="ringtest", seed=1, resolution=(w, h), maps=maps,
        physical={"radius_km": 60268.0, "height_scale": 0.004,
                  "height_midlevel": 0.5,
                  "ring_inner_km": 74500.0, "ring_outer_km": 136780.0},
        preset_doc={},
    )
    write_manifest(tmp_path, manifest)


def test_harsh_ring_strip_passes_validation(tmp_path):
    """A strip of hard tangentially-uniform ringlet edges (the worst case for
    the sphere checks) must PASS: rings gets finiteness only."""
    strip = np.zeros((128, 16, 4), dtype=np.float32)
    strip[::7] = 1.0  # hard radial ringlet edges, uniform across the strip
    _write_mapset(tmp_path, strip)
    report = validate_mapset(tmp_path)
    assert report.ok, report.summary()
    ring_checks = [c.name for c in report.checks if "rings" in c.name]
    assert ring_checks, "rings map must still be checked (finiteness)"
    assert all("finite" in n for n in ring_checks), ring_checks


def test_non_finite_ring_strip_fails_validation(tmp_path):
    strip = np.zeros((128, 16, 4), dtype=np.float32)
    strip[3, 2, 1] = np.nan
    _write_mapset(tmp_path, strip)
    report = validate_mapset(tmp_path)
    assert not report.ok
