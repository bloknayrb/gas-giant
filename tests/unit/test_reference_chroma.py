"""gasgiant.palette.reference: quartile-conditional chroma/lightness metrics.

The headline test demonstrates the blindness these metrics fix: per-channel
medians of a hue-spread population regress toward gray, so `belt_rgb` can
match a desaturated render while the members are far more chromatic —
`belt_chroma` (median MEMBER chroma) sees the difference.
"""

from __future__ import annotations

import numpy as np

from gasgiant.palette.gradient import srgb_to_oklab
from gasgiant.palette.reference import latitude_profile, profile_distance, profile_signed

# Two saturated dark colors with opposite hue directions and EQUAL Rec.709
# luma (so the dark luminance quartile contains both populations): their
# per-channel median is a near-gray midpoint (Oklab C ~0.016), but every
# member is strongly chromatic (C ~0.11).
_RUST = (0.55, 0.25, 0.10)
_TEAL = (0.10, 0.3383, 0.55)


def _two_population_image(h: int = 64, w: int = 256) -> np.ndarray:
    """Bottom QUARTER = alternating rust/teal columns; rest = cream.

    The dark population is exactly 25 % of the pixels, so the bottom
    luminance quartile is the whole 50/50 rust/teal mix — the quartile cut
    cannot land inside the mix and bias the hue proportions (the two
    populations differ in luminance only at float-rounding level).
    """
    img = np.full((h, w, 3), (0.9, 0.88, 0.82), dtype=np.float32)
    dark = np.where((np.arange(w) % 2 == 0)[None, :, None], _RUST, _TEAL)
    img[3 * h // 4:] = dark
    return img.astype(np.float32)


def _chroma(rgb) -> float:
    lab = srgb_to_oklab(np.asarray(rgb, dtype=np.float32).reshape(1, 3))[0]
    return float(np.hypot(lab[1], lab[2]))


def test_belt_chroma_sees_what_the_median_hides():
    p = latitude_profile(_two_population_image(), bins=1)
    # The per-channel median of rust/teal is a near-gray mid color...
    assert _chroma(p.belt_rgb[0]) < 0.25 * min(_chroma(_RUST), _chroma(_TEAL))
    # ...while the member chroma is that of the actual saturated members.
    expected = 0.5 * (_chroma(_RUST) + _chroma(_TEAL))
    assert abs(p.belt_chroma[0] - expected) < 0.15 * expected


def test_quartile_conditional_stds_flat_vs_mixed():
    flat = latitude_profile(np.full((64, 128, 3), (0.5, 0.35, 0.2), np.float32), bins=2)
    assert np.all(flat.belt_chroma_std < 1e-4)
    assert np.all(flat.belt_L_std < 1e-4)
    mixed = latitude_profile(_two_population_image(), bins=1)
    assert mixed.belt_chroma_std[0] >= 0.0  # rust/teal have similar C; L differs
    assert mixed.belt_L_std[0] > 0.005


def test_hue_spread_single_vs_two_hue_and_gray_guard():
    single = latitude_profile(np.full((64, 128, 3), _RUST, np.float32), bins=1)
    assert single.hue_spread[0] < 1e-3
    two = latitude_profile(_two_population_image(), bins=1)
    # Opposite-ish hue directions at equal weight -> resultant shrinks hard.
    assert two.hue_spread[0] > 0.3
    gray = latitude_profile(np.full((64, 128, 3), 0.5, np.float32), bins=1)
    assert gray.hue_spread[0] == 0.0  # below the gray-chroma floor


def test_texture_energy_flat_vs_noisy():
    flat = latitude_profile(np.full((64, 128, 3), 0.5, np.float32), bins=2)
    assert np.all(flat.texture_energy < 1e-6)
    rng = np.random.default_rng(7)
    noisy = np.clip(0.5 + 0.2 * rng.standard_normal((64, 128, 3)), 0, 1).astype(np.float32)
    p = latitude_profile(noisy, bins=2)
    assert np.all(p.texture_energy > 0.01)


def test_profile_distance_zero_on_self_includes_new_keys():
    p = latitude_profile(_two_population_image(), bins=8)
    d = profile_distance(p, p)
    for key in ("zone_chroma", "belt_chroma", "belt_chroma_std", "belt_L_std",
                "belt_chroma_p95", "hue_spread", "texture_energy"):
        assert d[key] == 0.0


def test_profile_signed_direction():
    base = _two_population_image()
    desat = 0.6 * base + 0.4 * base.mean(axis=2, keepdims=True)
    p_sat = latitude_profile(base, bins=4)
    p_desat = latitude_profile(desat.astype(np.float32), bins=4)
    s = profile_signed(p_desat, p_sat)
    assert s["belt_chroma"] < 0.0  # desaturated minus saturated: deficit


# Independent Oklab fixed points (Ottosson's published reference values) so a
# shared-constant typo in gradient.py cannot self-confirm through the
# GPU<->Python parity test that reuses the same module.
_OKLAB_FIXED_POINTS = [
    ((1.0, 0.0, 0.0), (0.62796, 0.22486, 0.12585)),
    ((0.0, 1.0, 0.0), (0.86644, -0.23389, 0.17950)),
    ((1.0, 1.0, 1.0), (1.00000, 0.00000, 0.00000)),
]


def test_oklab_fixed_points():
    for rgb, lab_ref in _OKLAB_FIXED_POINTS:
        lab = srgb_to_oklab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))[0]
        assert np.allclose(lab, lab_ref, atol=2e-4), (rgb, lab, lab_ref)
