"""Unit tests for scripts/measure_grs.py (P4/P5 GRS measurement suite).

Pure synthetic numpy — NO gpu mark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# scripts/ is not a package; add its parent to sys.path so we can import directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from measure_grs import fit_ellipse_aspect, ring_closure, ring_ripple_std

# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

_SIZE = 400
_CX = 200.0
_CY = 200.0
_RC = 60.0
_ASPECT = 1.0  # circular for simplicity in most tests


def _make_coords(size: int, cx: float, cy: float, rc: float, aspect: float = 1.0):
    """Return (q, theta) arrays for a pixel grid."""
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = (cols - cx) / (rc * aspect)
    dy = (rows - cy) / rc
    q = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return q, theta


# ---------------------------------------------------------------------------
# Test 1: ring_ripple_std is high for concentric, low for azimuthally modulated
# ---------------------------------------------------------------------------

class TestRippleMetric:
    def test_ripple_metric_high_for_concentric_low_for_modulated(self):
        """Concentric ring (theta-independent) → HIGH std.
        Azimuthally modulated ring (P4 fix) → LOW std, much smaller."""
        size = _SIZE
        q, theta = _make_coords(size, _CX, _CY, _RC, _ASPECT)

        # Concentric: L = 0.5 + 0.1*cos(q*28), theta-independent.
        L_conc = (0.5 + 0.1 * np.cos(q * 28.0)).astype(np.float32)

        # Modulated (P4 fix): L = 0.5 + 0.1*cos(q*28 + 5*theta).
        # Azimuthal mean for each q: E_theta[cos(q*28 + 5*theta)] = 0
        # because integral of cos(q*28 + 5*theta) over theta = 0 (integer m=5≠0).
        L_mod = (0.5 + 0.1 * np.cos(q * 28.0 + 5.0 * theta)).astype(np.float32)

        std_conc = ring_ripple_std(L_conc, _CX, _CY, _RC, _ASPECT)
        std_mod = ring_ripple_std(L_mod, _CX, _CY, _RC, _ASPECT)

        # Concentric rings should register a clear signal.
        assert std_conc > 0.03, (
            f"Expected high ripple_std for concentric rings, got {std_conc:.4f}"
        )
        # Modulated rings should have much lower std (azimuthal mean ≈ 0).
        assert std_mod < 0.3 * std_conc, (
            f"Expected std_mod ({std_mod:.4f}) < 0.3 * std_conc ({std_conc:.4f})"
        )
        # Sanity: both are non-negative
        assert std_conc >= 0.0
        assert std_mod >= 0.0


# ---------------------------------------------------------------------------
# Test 2: fit_ellipse_aspect recovers a known aspect ratio
# ---------------------------------------------------------------------------

class TestEllipseFit:
    def test_ellipse_fit_recovers_aspect(self):
        """Dark elliptical ring with known aspect 2.0 → recovered aspect ≈ 2.0."""
        size = 400
        cx, cy = 200.0, 200.0
        rc = 60.0
        true_aspect = 2.0

        rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
        # Elliptical q in the spot's own frame
        dx = (cols - cx) / (rc * true_aspect)
        dy = (rows - cy) / rc
        q_ell = np.hypot(dx, dy)

        # Dark ring: L is low near q_ell ≈ 1.0 (annulus 0.85 – 1.15)
        L = np.ones((size, size), dtype=np.float32) * 0.7
        ring_mask = (q_ell >= 0.85) & (q_ell <= 1.15)
        L[ring_mask] = 0.1  # clearly dark

        result = fit_ellipse_aspect(L, cx, cy, rc, search=2.2)
        assert result is not None, "fit_ellipse_aspect returned None — too few dark points"
        # Allow ±20% tolerance on the recovered aspect ratio
        assert abs(result - true_aspect) / true_aspect <= 0.20, (
            f"Expected aspect ≈ {true_aspect}, got {result:.3f} "
            f"(deviation {abs(result - true_aspect) / true_aspect:.1%})"
        )


# ---------------------------------------------------------------------------
# Test 3: ring_closure detects a gap correctly
# ---------------------------------------------------------------------------

class TestRingClosure:
    def _make_ring_image(self, size: int, cx: float, cy: float, rc: float,
                         gap_start_deg: float = None, gap_end_deg: float = None,
                         aspect: float = 1.0) -> np.ndarray:
        """Create a synthetic image with a dark elliptical ring.

        If gap_start_deg/gap_end_deg are provided, the ring is bright (no dark
        dip) in that angular wedge.
        """
        rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
        dx = (cols - cx) / (rc * aspect)
        dy = (rows - cy) / rc
        q_ell = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)  # [-pi, pi]

        # Background luminance
        L = np.ones((size, size), dtype=np.float32) * 0.6
        # Dark ring: near q_ell ≈ 1
        ring_mask = (q_ell >= 0.80) & (q_ell <= 1.20)
        L[ring_mask] = 0.1

        if gap_start_deg is not None and gap_end_deg is not None:
            # Remove the dark ring in the gap wedge (replace with bright)
            start_rad = np.radians(gap_start_deg)
            end_rad = np.radians(gap_end_deg)
            # Normalize both to [0, 2pi)
            th_pos = (theta + 2 * np.pi) % (2 * np.pi)
            s_pos = start_rad % (2 * np.pi)
            e_pos = end_rad % (2 * np.pi)
            if s_pos <= e_pos:
                gap_mask = (th_pos >= s_pos) & (th_pos <= e_pos)
            else:
                gap_mask = (th_pos >= s_pos) | (th_pos <= e_pos)
            L[ring_mask & gap_mask] = 0.6  # restore to background (no contrast, no ring)

        return L.astype(np.float32)

    def test_ring_closure_detects_gap(self):
        """Full ring → small max_gap; ring with ~90° gap → max_gap ≈ 90°."""
        size = 400
        cx, cy = 200.0, 200.0
        rc = 60.0
        aspect = 1.0

        # Full ring: no gap
        L_full = self._make_ring_image(size, cx, cy, rc, aspect=aspect)
        gap_deg_full, ratio_full = ring_closure(L_full, cx, cy, rc, aspect)

        # Ring with 90° gap (0° to 90°)
        L_gap = self._make_ring_image(size, cx, cy, rc,
                                      gap_start_deg=0.0, gap_end_deg=90.0,
                                      aspect=aspect)
        gap_deg_gap, ratio_gap = ring_closure(L_gap, cx, cy, rc, aspect)

        # Full ring: gap should be very small (< 20°)
        assert gap_deg_full < 20.0, (
            f"Full ring should have small gap, got {gap_deg_full:.1f}°"
        )

        # Gapped ring: max_gap should be significantly larger (~90°)
        assert gap_deg_gap > 60.0, (
            f"Ring with 90° gap should have large max_gap, got {gap_deg_gap:.1f}°"
        )

        # The gapped ring has a larger gap than the full ring
        assert gap_deg_gap > gap_deg_full + 40.0, (
            f"Gap metric should distinguish full ({gap_deg_full:.1f}°) "
            f"vs gapped ({gap_deg_gap:.1f}°) ring"
        )

        # Full ring depth_ratio should be reasonable (≥ 0.40)
        assert ratio_full >= 0.40, (
            f"Full ring depth_ratio should be ≥ 0.40, got {ratio_full:.3f}"
        )
