"""band_template_arrays: band-skeleton extraction from synthetic images."""

from __future__ import annotations

import numpy as np

from gasgiant.palette.reference import band_template_arrays


def _striped(edges_deg, levels, h=720, w=24, polar=0.45):
    """Equirect (h, w, 3) image of constant-luminance latitude bands; the
    polar caps (outside the first/last edge) sit at a mid gray."""
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    img = np.full((h, w, 3), polar, dtype=np.float32)
    for top, bot, level in zip(edges_deg[:-1], edges_deg[1:], levels, strict=False):
        img[(lat <= top) & (lat > bot)] = level
    return img


def _alternates(mask: np.ndarray) -> bool:
    return bool(np.all(mask[1:] != mask[:-1]))


def test_recovers_synthetic_bands():
    true_edges = [76.0, 48.0, 22.0, 0.0, -24.0, -50.0, -76.0]
    levels = [0.75, 0.30, 0.75, 0.30, 0.75, 0.30]
    out = band_template_arrays(_striped(true_edges, levels))
    assert len(out["band_lum"]) == 6
    assert out["edges_deg"][0] == 90.0 and out["edges_deg"][-1] == -90.0
    assert _alternates(out["is_zone"])
    assert out["is_zone"][0]  # first band is the bright one
    np.testing.assert_allclose(out["edges_deg"][1:-1], true_edges[1:-1], atol=2.5)
    # Band luminances recovered near the painted levels.
    np.testing.assert_allclose(out["band_lum"][out["is_zone"]], 0.75, atol=0.05)
    np.testing.assert_allclose(out["band_lum"][~out["is_zone"]], 0.30, atol=0.05)


def test_narrow_band_merged_away():
    # A 1.5-deg bright sliver inside a belt is below min_width_deg and must
    # be folded back into its surroundings (3-way merge keeps alternation).
    true_edges = [76.0, 40.0, 6.0, 4.5, -30.0, -76.0]
    levels = [0.75, 0.30, 0.78, 0.30, 0.75]
    out = band_template_arrays(_striped(true_edges, levels))
    widths = -np.diff(out["edges_deg"][1:-1])
    assert np.all(widths >= 2.0)
    assert len(out["band_lum"]) % 2 == 0
    assert _alternates(out["is_zone"])


def test_odd_count_forced_even_by_edge_merge():
    true_edges = [76.0, 50.0, 28.0, 8.0, -12.0, -32.0, -52.0, -76.0]
    levels = [0.75, 0.30, 0.75, 0.30, 0.75, 0.30, 0.75]
    out = band_template_arrays(_striped(true_edges, levels))
    assert len(out["band_lum"]) % 2 == 0
    assert _alternates(out["is_zone"])
    # The interior structure survives: at least the 4 strongest inner edges.
    assert len(out["edges_deg"]) >= 6


def test_max_bands_cap():
    edges = list(np.linspace(76.0, -76.0, 13))  # 12 bands
    levels = [0.75 if i % 2 == 0 else 0.30 for i in range(12)]
    out = band_template_arrays(_striped(edges, levels), max_bands=8)
    assert len(out["band_lum"]) <= 8
    assert len(out["band_lum"]) % 2 == 0
    assert _alternates(out["is_zone"])
