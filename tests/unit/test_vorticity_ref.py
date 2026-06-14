"""Formula-lock tests for gasgiant.sim.vorticity_ref (P2, v1.6).

These tests pin the SIGN CONVENTION and METRIC of the vorticity operators
so that any accidental formula change will cause a hard failure.

GPU is NOT required — all tests are pure numpy.

Contents
--------
(a) Spherical-harmonic eigenvalue tests (sign/metric lock)
    - Zonal  Y ∝ (3 sin²φ − 1),      eigenvalue −6  (tests tanφ sign)
    - Non-zonal  Y ∝ cos²φ · cos 2λ, eigenvalue −6  (tests 1/cos²φ λ-term)

(b) Gaussian-vortex round-trip:
    Compute laplacian_sphere(ψ_G) numerically and compare to vortex_omega_ref,
    at a small r_core (~0.05 rad) AND a GRS-scale r_core (~0.15 rad).
    Also verify that omitting the cot(d) term gives a larger residual (term earns
    its place).

(c) jet_vorticity: integrate a test u, broadcast to 2-D ψ_jet, compare
    jet_vorticity(u, lat) to a 1-D slice of laplacian_sphere(ψ_jet_2d).
"""

from __future__ import annotations

import importlib.resources as ir

import numpy as np
import pytest

from gasgiant.sim.vorticity_ref import (
    coriolis,
    jet_vorticity,
    laplacian_patch,
    laplacian_sphere,
    vortex_omega_ref,
)


def test_laplacian_patch_spherical_harmonic_eigenvalues():
    """v1.6 Phase B (P6): the azimuthal-equidistant patch Laplacian recovers
    ∇²Y_l^m = −l(l+1)·Y on a polar patch, for BOTH a zonal and an azimuthal
    harmonic — locking the AE-metric coefficients + the pole regularity."""
    n = 256
    rho_max = np.radians(34.0)
    idx = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    s = (idx * rho_max)[np.newaxis, :] * np.ones((n, 1))
    t = (idx * rho_max)[:, np.newaxis] * np.ones((1, n))
    rho = np.hypot(s, t)
    theta = np.arctan2(t, s)
    interior = (rho < np.radians(30.0)) & (rho > np.radians(3.0))

    # Zonal Y_2^0 ∝ 3cos²ρ − 1  (tests the radial/cot term + pole regularity).
    y0 = 3.0 * np.cos(rho) ** 2 - 1.0
    ev0 = np.median((laplacian_patch(y0, rho_max) / y0)[interior & (np.abs(y0) > 0.3)])
    assert abs(ev0 - (-6.0)) < 0.3, f"zonal eigenvalue {ev0:.3f} != -6"

    # Azimuthal Y_2^2 ∝ sin²ρ·cos(2θ)  (tests the 1/sin²ρ azimuthal term).
    y2 = np.sin(rho) ** 2 * np.cos(2.0 * theta)
    m2 = interior & (np.abs(y2) > 0.05)
    ev2 = np.median((laplacian_patch(y2, rho_max) / y2)[m2])
    assert abs(ev2 - (-6.0)) < 0.3, f"azimuthal eigenvalue {ev2:.3f} != -6"

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _grid(H: int = 256, W: int = 512):
    """Return (lat, lon, LAT2D, LON2D) for an H×W equirectangular grid."""
    lat = np.linspace(-np.pi / 2, np.pi / 2, H)   # ascending
    lon = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    LAT2D, LON2D = np.meshgrid(lat, lon, indexing="ij")
    return lat, lon, LAT2D, LON2D


# ---------------------------------------------------------------------------
# (a) Spherical-harmonic eigenvalue tests
# ---------------------------------------------------------------------------

class TestSHEigenvalues:
    """∇²Y_l^m = −l(l+1)·Y_l^m.  Both test cases use l=2, eigenvalue = −6."""

    # Restrict to |lat| < 70° to avoid the cosφ-floor distortion near poles.
    LAT_MASK_DEG = 70.0

    def _lat_mask(self, lat: np.ndarray) -> np.ndarray:
        return np.abs(lat) < np.radians(self.LAT_MASK_DEG)

    def test_zonal_Y20_eigenvalue(self):
        """Y_2^0 ∝ (3 sin²φ − 1).  ∇²Y = −6·Y.

        This is the critical SIGN TEST for the tanφ term: if tanφ has the
        wrong sign the eigenvalue ratio flips sign or the residual blows up.
        """
        H, W = 256, 512
        lat, _, LAT2D, _ = _grid(H, W)

        Y = 3.0 * np.sin(LAT2D) ** 2 - 1.0   # Y_2^0 (unnormalised)
        lap = laplacian_sphere(Y, lat)

        # Expected: ∇²Y = −6·Y
        ratio = lap / Y
        mask = self._lat_mask(lat)  # (H,) — applied row-wise
        ratio_masked = ratio[mask, :]   # select rows in latitude range

        measured_eigenvalue = np.median(ratio_masked)
        error = abs(measured_eigenvalue - (-6.0)) / 6.0

        print(f"\nY_2^0 (zonal): measured eigenvalue = {measured_eigenvalue:.4f}, "
              f"expected -6.000, relative error = {error:.4f}")

        assert error < 0.05, (
            f"Zonal Y_2^0 eigenvalue error {error:.4%} exceeds 5% — "
            f"check tanφ sign in laplacian_sphere"
        )

    def test_nonzonal_Y22_eigenvalue(self):
        """Y_2^2 ∝ cos²φ · cos 2λ.  ∇²Y = −6·Y.

        Tests the 1/cos²φ λ-term (the zonal test does not excite it).
        """
        H, W = 256, 512
        lat, lon, LAT2D, LON2D = _grid(H, W)

        Y = np.cos(LAT2D) ** 2 * np.cos(2.0 * LON2D)   # Y_2^2 (unnormalised)
        lap = laplacian_sphere(Y, lat)

        ratio = lap / np.where(np.abs(Y) > 1e-8, Y, np.nan)
        mask = self._lat_mask(lat)
        ratio_masked = ratio[mask, :]
        ratio_masked = ratio_masked[np.isfinite(ratio_masked)]

        measured_eigenvalue = np.median(ratio_masked)
        error = abs(measured_eigenvalue - (-6.0)) / 6.0

        print(f"\nY_2^2 (non-zonal): measured eigenvalue = {measured_eigenvalue:.4f}, "
              f"expected -6.000, relative error = {error:.4f}")

        assert error < 0.05, (
            f"Non-zonal Y_2^2 eigenvalue error {error:.4%} exceeds 5% — "
            f"check 1/cos²φ term in laplacian_sphere"
        )


# ---------------------------------------------------------------------------
# (b) Gaussian-vortex round-trip
# ---------------------------------------------------------------------------

def _sphere_xyz(lat0: float, lon0: float) -> np.ndarray:
    """Unit sphere position for a vortex centre."""
    return np.array([
        np.cos(lat0) * np.cos(lon0),
        np.sin(lat0),
        np.cos(lat0) * np.sin(lon0),
    ])


def _great_circle_d(LAT2D, LON2D, lat0, lon0):
    """Great-circle distance array from (lat0, lon0)."""
    p = _sphere_xyz(lat0, lon0)
    px = np.cos(LAT2D) * np.cos(LON2D)
    py = np.sin(LAT2D)
    pz = np.cos(LAT2D) * np.sin(LON2D)
    dot = np.clip(px * p[0] + py * p[1] + pz * p[2], -1.0, 1.0)
    return np.arccos(dot)


def _omega_no_cot(S, r_core, d):
    """vortex_omega_ref WITHOUT the cot(d)/d·tan(d) correction term (for comparison).

    Omits the geodesic-metric term  −(2S/r²)·(d/tan d)·exp(−q²),
    keeping only the radial curvature part  (S/r²)·(4q²−2)·exp(−q²).
    """
    d = np.asarray(d, dtype=float)
    q = d / r_core
    exp_q2 = np.exp(-q**2)
    return (S / r_core**2) * (4.0 * q**2 - 2.0) * exp_q2


class TestVortexRoundTrip:
    """Numerically verify ∇²ψ_G ≈ vortex_omega_ref across r_core scales."""

    H, W = 256, 512
    # Vortex centre away from poles and away from the lon=0/2π seam.
    LAT0 = np.radians(20.0)
    LON0 = np.radians(90.0)
    S = 1.0

    def _run(self, r_core: float, atol_frac: float):
        """Return (max_rel_error_with_cot, max_rel_error_no_cot) in the core."""
        lat, lon, LAT2D, LON2D = _grid(self.H, self.W)
        d = _great_circle_d(LAT2D, LON2D, self.LAT0, self.LON0)
        q = d / r_core

        psi = self.S * np.exp(-q**2)
        lap_num = laplacian_sphere(psi, lat)
        omega_ref = vortex_omega_ref(self.S, r_core, d)
        omega_nocot = _omega_no_cot(self.S, r_core, d)

        # Evaluate in the core region (q < 2, away from noise-dominated far field)
        core_mask = q < 2.0
        # Also exclude near-poles (cosφ floor noise)
        pole_mask = np.abs(lat) < np.radians(75.0)
        mask = core_mask & pole_mask[:, np.newaxis]

        scale = np.max(np.abs(omega_ref[mask]))

        err_with = np.max(np.abs((lap_num - omega_ref)[mask])) / scale
        err_no   = np.max(np.abs((lap_num - omega_nocot)[mask])) / scale

        return err_with, err_no, scale

    def test_small_r_core(self):
        """Small r_core (~0.05 rad): tight round-trip, cot term is minor."""
        r_core = 0.05
        err_with, err_no, scale = self._run(r_core, atol_frac=0.05)

        print(f"\nSmall r_core={r_core:.3f} rad:"
              f"  err_with_cot={err_with:.4f}  err_no_cot={err_no:.4f}  scale={scale:.4f}")

        assert err_with < 0.05, (
            f"Small r_core round-trip error {err_with:.4%} > 5% — "
            f"check vortex_omega_ref or laplacian_sphere"
        )

    def test_grs_r_core(self):
        """GRS r_core (~0.15 rad): cot(d) term must reduce residual."""
        r_core = 0.15
        err_with, err_no, scale = self._run(r_core, atol_frac=0.10)

        print(f"\nGRS r_core={r_core:.3f} rad:"
              f"  err_with_cot={err_with:.4f}  err_no_cot={err_no:.4f}  scale={scale:.4f}")

        assert err_with < 0.10, (
            f"GRS r_core round-trip error {err_with:.4%} > 10% — "
            f"check vortex_omega_ref or laplacian_sphere"
        )
        # Verify the cot term earns its place: no-cot residual must be worse.
        assert err_no > err_with, (
            f"cot(d) term did NOT reduce residual at GRS scale: "
            f"err_with_cot={err_with:.4f}, err_no_cot={err_no:.4f}. "
            f"The term is required — check formula."
        )

    def test_grs_cot_term_necessary(self):
        """Without the cot(d) term, the GRS residual is meaningfully larger."""
        r_core = 0.15
        err_with, err_no, _ = self._run(r_core, atol_frac=0.10)
        # The no-cot error must be at least 50% worse (relative to err_with).
        margin = (err_no - err_with) / max(err_with, 1e-10)
        print(f"\nGRS cot necessity: err_with={err_with:.4f}  err_no={err_no:.4f}  "
              f"margin={margin:.2f}x")
        assert margin > 0.5, (
            f"cot term margin {margin:.2f}x < 0.5x — "
            f"term may be unnecessary or incorrectly implemented"
        )


# ---------------------------------------------------------------------------
# (c) jet_vorticity
# ---------------------------------------------------------------------------

class TestJetVorticity:
    """jet_vorticity(u, lat) should match ∇²ψ_jet for zonal flow."""

    def test_jet_matches_laplacian(self):
        """Build ψ_jet from a test u, verify jet_vorticity ≈ laplacian_sphere slice."""
        H, W = 256, 512
        lat = np.linspace(-np.pi / 2, np.pi / 2, H)

        # Smooth zonal wind profile: a Gaussian jet centred at 20° N.
        lat0 = np.radians(20.0)
        sigma = np.radians(15.0)
        u = np.exp(-((lat - lat0) / sigma) ** 2)   # (H,)

        # Integrate u = −dψ/dφ  ⟹  ψ(φ) = −∫u dφ
        # Use cumulative trapezoid (or cumsum approximation).
        dphi = lat[1] - lat[0]
        psi_1d = -np.cumsum(u) * dphi   # approximate indefinite integral
        # Broadcast to 2-D (uniform longitude — should be a pure zonal function)
        psi_2d = np.tile(psi_1d[:, np.newaxis], (1, W))

        # Numerical Laplacian and reference jet_vorticity
        lap = laplacian_sphere(psi_2d, lat)
        zeta_jet = jet_vorticity(u, lat)

        # Compare at interior latitudes (avoid poles and near-boundary rows)
        mask = np.abs(lat) < np.radians(70.0)
        lap_slice = np.mean(lap[mask, :], axis=1)   # longitude-average (should be uniform)
        ref_slice = zeta_jet[mask]

        scale = np.max(np.abs(ref_slice))
        rel_err = np.max(np.abs(lap_slice - ref_slice)) / max(scale, 1e-10)

        print(f"\njet_vorticity round-trip: rel_err={rel_err:.4f}  scale={scale:.4e}")

        assert rel_err < 0.05, (
            f"jet_vorticity vs laplacian_sphere discrepancy {rel_err:.4%} > 5%"
        )


# ---------------------------------------------------------------------------
# (d) Coriolis sanity
# ---------------------------------------------------------------------------

class TestCoriolis:
    def test_equator_zero(self):
        assert coriolis(0.0, 1.0) == pytest.approx(0.0)

    def test_pole_equals_f0(self):
        assert coriolis(np.pi / 2, 2.5) == pytest.approx(2.5)

    def test_southern_hemisphere_negative(self):
        f = coriolis(np.radians(-30.0), 1.0)
        assert f < 0.0

    def test_array_input(self):
        lat = np.array([0.0, np.pi / 6, np.pi / 2])
        f = coriolis(lat, 1.0)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# (e) GLSL mirror assertion (GPU lock deferred to P3)
# ---------------------------------------------------------------------------

class TestGLSLMirror:
    """Assert that vortex_omega.glsl textually mirrors vorticity_ref.py.

    A full GPU round-trip is deferred to P3's omega kernel dispatch.
    Here we do a structural (textual) audit of the GLSL source.
    """

    def _glsl_text(self) -> str:
        pkg = "gasgiant.sim.kernels"
        return ir.files(pkg).joinpath("vortex_omega.glsl").read_text(encoding="utf-8")

    def test_glsl_file_exists(self):
        text = self._glsl_text()
        assert len(text) > 100, "vortex_omega.glsl appears empty"

    def test_glsl_has_d_over_tand_fallback(self):
        """GLSL must implement the small-d series fallback for d/tan(d)."""
        text = self._glsl_text()
        # Expect the series approximation  1 − d²/3
        assert "d * d / 3.0" in text or "d*d/3.0" in text or "d / 3.0" in text, (
            "GLSL missing d/tan(d) small-d series  '1 − d²/3'"
        )

    def test_glsl_sign_convention_comment(self):
        """GLSL file must document the ω = +∇²ψ sign convention."""
        text = self._glsl_text()
        assert "+∇²ψ" in text or "omega = +lap" in text or "+nabla" in text.lower() or \
               "ω = +∇²ψ" in text or "omega = +grad" in text or "= +lap" in text or \
               "sign convention" in text.lower(), (
            "GLSL missing sign-convention comment (ω = +∇²ψ)"
        )

    def test_glsl_has_term1_and_term2(self):
        """GLSL must have both ω terms matching vorticity_ref.py."""
        text = self._glsl_text()
        assert "term1" in text and "term2" in text, (
            "GLSL missing term1/term2 structure mirroring vortex_omega_ref"
        )

    def test_glsl_uses_magnitude_cull(self):
        """GLSL must cull by magnitude, not a fixed q threshold."""
        text = self._glsl_text()
        assert "abs(contrib)" in text and "abs(scale)" in text, (
            "GLSL missing magnitude-based cull (should use abs(contrib) vs abs(scale))"
        )

    def test_glsl_has_coriolis_helper(self):
        """GLSL must define the coriolis() helper."""
        text = self._glsl_text()
        assert "float coriolis(" in text, (
            "GLSL missing coriolis() helper function"
        )

    def test_glsl_has_great_circle_distance(self):
        """GLSL must compute great-circle distance via acos(dot(...))."""
        text = self._glsl_text()
        assert "acos(" in text and "dot(p," in text, (
            "GLSL missing great-circle distance computation"
        )

    def test_glsl_hero_aspect_branch(self):
        """GLSL must replicate the hero-aspect q computation from vortex_stamp.glsl."""
        text = self._glsl_text()
        assert "asp == 1.0" in text or "asp==" in text.replace(" ", ""), (
            "GLSL missing aspect-branch (asp == 1.0 round vs elliptical q)"
        )
