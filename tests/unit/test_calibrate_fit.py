"""gasgiant.palette.reference: chroma-restored calibration fits."""

from __future__ import annotations

import numpy as np

from gasgiant.palette.gradient import srgb_to_oklab
from gasgiant.palette.reference import anchor_fit, chroma_restored_rgb

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Equal-luma rust/teal (see test_reference_chroma): the per-channel median
# of the pair is near-gray while every member is chromatic.
_RUST = np.array([0.55, 0.25, 0.10], dtype=np.float32)
_TEAL = np.array([0.10, 0.3383, 0.55], dtype=np.float32)


def _chroma(rgb) -> float:
    lab = srgb_to_oklab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))[0]
    return float(np.hypot(lab[1], lab[2]))


def test_median_desaturates_and_chroma_restore_recovers():
    rgb = np.concatenate([np.tile(_RUST, (500, 1)), np.tile(_TEAL, (500, 1))])
    lum = rgb @ _LUMA
    median_fit = np.median(rgb, axis=0)
    restored = chroma_restored_rgb(rgb, lum, 0.0, 1.0, chroma_pct=0.6)
    members = sorted([_chroma(_RUST), _chroma(_TEAL)])
    expected_c = np.quantile([_chroma(_RUST)] * 500 + [_chroma(_TEAL)] * 500, 0.6)
    assert _chroma(median_fit) < 0.3 * members[0]          # the defect
    assert abs(_chroma(restored) - expected_c) < 0.25 * expected_c  # the fix


def test_chroma_restore_resists_chromatic_minority():
    """A 15% festoon-blue minority inside a rust belt must not flip the
    fitted hue (the top-chroma-sub-quartile alternative would latch on)."""
    rgb = np.concatenate([np.tile(_RUST, (850, 1)), np.tile(_TEAL, (150, 1))])
    lum = rgb @ _LUMA
    restored = chroma_restored_rgb(rgb, lum, 0.0, 1.0)
    lab = srgb_to_oklab(restored.reshape(1, 3))[0]
    rust_lab = srgb_to_oklab(_RUST.reshape(1, 3).astype(np.float64))[0]
    # Same hue quadrant as the rust majority (positive a and b).
    assert np.sign(lab[1]) == np.sign(rust_lab[1])
    assert np.sign(lab[2]) == np.sign(rust_lab[2])


def test_low_chroma_fallback_returns_plain_median():
    gray = np.full((400, 3), 0.5, dtype=np.float32)
    gray[::7] += 0.002  # sub-floor chroma noise
    lum = gray @ _LUMA
    out = chroma_restored_rgb(gray, lum, 0.0, 1.0)
    np.testing.assert_allclose(out, np.median(gray, axis=0), atol=1e-6)


def _banded_image(h: int = 90, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        img[row] = (0.9, 0.85, 0.7) if (row // 5) % 2 == 0 else (0.45, 0.3, 0.15)
    return img


def test_anchor_fit_five_stops_monotonic_in_luminance():
    img = _banded_image()
    for mode in ("median", "chroma-restore"):
        fitted = anchor_fit(img, 0.0, 12.0, mode=mode, stops=5)
        assert [pos for pos, _ in fitted] == [0.0, 0.25, 0.5, 0.75, 1.0]
        lums = [float(np.asarray(c, dtype=np.float32) @ _LUMA) for _, c in fitted]
        assert all(b >= a - 1e-4 for a, b in zip(lums, lums[1:], strict=False))


def test_anchor_fit_median_mode_matches_quartile_medians():
    img = _banded_image()
    fitted = dict(anchor_fit(img, 0.0, 12.0, mode="median", stops=3))
    np.testing.assert_allclose(fitted[0.0], (0.45, 0.3, 0.15), atol=0.02)
    np.testing.assert_allclose(fitted[1.0], (0.9, 0.85, 0.7), atol=0.02)


def test_anchor_fit_empty_window_falls_back_to_nearest_rows():
    img = _banded_image(h=10)
    fitted = anchor_fit(img, 89.0, 0.001, mode="median", stops=3)
    assert len(fitted) == 3
    assert all(np.all(np.isfinite(c)) for _, c in fitted)
